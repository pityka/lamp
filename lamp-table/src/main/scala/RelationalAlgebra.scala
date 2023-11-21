package lamp.table

import org.saddle._
import lamp._
import org.saddle.index.OuterJoin
import cats.effect.IO
import cats.effect.kernel.Resource
// import org.saddle.index.InnerJoin
// import org.saddle.index.JoinType

trait RelationalAlgebra { self: Table =>
  def resolveColumnIdx(cs: ColumnSelection) = cs.e match {
    case Left(value)  => colNames.getFirst(value)
    case Right(value) => value
  }
  // projection

  def colAt(idx: Int): Column = columns(idx)
  def apply(idx: Int): Column = columns(idx)
  def colsAt(idx: Int*): Table = colAt(idx: _*)
  def colAt(idx: Int*): Table =
    Table(idx.map(i => columns(i)).toVector, colNames.at(idx: _*))

  def firstCol(name: String): Column = colAt(colNames.getFirst(name))
  def col(name: String): Column = firstCol(name)
  def apply(name: String): Column = col(name)

  def cols(names: String*): Table =
    Table(
      colNames.apply(names: _*).map(i => columns(i)).toVector,
      colNames.at(colNames(names: _*))
    )
  def project(name: String*) = cols(name: _*)

  def withoutCol(s: Set[Int]) = {
    colAt(columns.zipWithIndex.map(_._2).filterNot(s.contains): _*)
  }
  def withoutCol(s: Int): Table = withoutCol(Set(s))
  def remove(s: Int*) = withoutCol(s.toSet)
  def remove(s: Int) = withoutCol(s)
  def remove(s: String) = withoutCol(colNames.get(s).toSet)

  def rename(i: Int, s: String) = withColNames(
    self.colNames.toSeq.updated(i, s).toIndex
  )

  // join

  def join(
      other: Table,
      chunkSize: Int = 5000,
      parallelism: Int = 8
  )(theta: Table => Column)(implicit scope: Scope): Table = {
    import cats.effect.unsafe.implicits.global
    Scope
      .bracket { implicit scope: Scope =>
        this.chunkedProduct(other, chunkSize).use { chunks =>
          IO.parTraverseN(parallelism)(chunks.map(_.allocated)) { alloc =>
            alloc.flatMap { case (table, release) =>
              IO(table.filter(theta(table))) <* release
            }
          }.map { t => Table.union(t: _*) }
        }
      }
      .unsafeRunSync()

  }

  def equijoin(
      col: ColumnSelection,
      other: Column,
      how: org.saddle.index.JoinType
  )(implicit scope: Scope): Table =
    this.equijoin(resolveColumnIdx(col), other.table, 0, how)

  def equijoin[IndexType](
      col: ColumnSelection,
      other: Table,
      otherCol: ColumnSelection,
      how: org.saddle.index.JoinType
  )(implicit scope: Scope): Table = {
    val col1 = resolveColumnIdx(col)
    val col2 = other.resolveColumnIdx(otherCol)
    val indexA = indexed(List(col1)).columns(col1).indexAs[IndexType].get.index
    val indexB = other
      .indexed(List(col2))
      .columns(col2)
      .indexAs[IndexType]
      .get
      .index
    val reindexer = indexA.join(indexB, how)

    val asub = reindexer.lTake.map(i => this.rows(i)).getOrElse(this)
    val bsub = reindexer.rTake
      .map(i => other.rows(i))
      .getOrElse(other)

    val a: Table =
      if (how == org.saddle.index.RightJoin)
        asub.withoutCol(Set(col1))
      else asub
    val b: Table =
      if (
        how == org.saddle.index.LeftJoin || how == org.saddle.index.InnerJoin || how == org.saddle.index.OuterJoin
      )
        bsub.withoutCol(Set(col2))
      else bsub

    val bind = a.bind(b)

    if (how == OuterJoin) {

      val kA = asub.columns(col1).values
      val kB = bsub.columns(col2).values

      val idx =
        reindexer.rTake.getOrElse(array.range(0, other.numRows.toInt)).toVec

      val nonmissingIdxLocationV = idx.find(_ >= 0L)
      val nonmissingIdxLocation =
        lamp.saddle
          .fromLongVec(nonmissingIdxLocationV.map(_.toLong), device)
          .view(
            nonmissingIdxLocationV.length.toLong :: kB.shape
              .drop(1)
              .map(_ => 1L): _*
          )
          .expand(
            nonmissingIdxLocationV.length.toLong :: kB.shape.drop(1)
          )
      val nonmissingIdxValueV = idx.take(nonmissingIdxLocationV.toArray)
      val nonmissingIdxValue =
        lamp.saddle.fromLongVec(nonmissingIdxValueV.map(_.toLong), device)

      val nonmissingValues =
        other.colAt(col2).values.indexSelect(0, nonmissingIdxValue)

      val ret = kA.scatter(0, nonmissingIdxLocation, nonmissingValues)
      val merged = asub.columns(col1).copy(values = ret)
      bind.replaceCol(col1, merged)

    } else bind

  }

  // set operations

  def union[S: Sc](others: Table*): Table = {
    require((colNames +: others.map(_.colNames)).distinct.size == 1)
    val c = (0 until numCols).map { colIdx =>
      val tpe = colType(colIdx)
      val s1 = colAt(colIdx).values
      val s3 = STen.cat(List(s1) ++ others.map(_.colAt(colIdx).values), dim = 0)
      Column(s3, tpe, None)
    }.toVector
    Table(c, colNames)
  }

  // extension

  def bind(other: Table): Table = {
    require(numRows == other.numRows)
    val c = columns ++ other.columns
    Table(c, colNames.concat(other.colNames))
  }

  def bind(col: Column): Table = bind(col.table)
  def bindWithName(col: Column, name: String): Table = bind(
    col.tableWithName(name)
  )
  def extend(fn: Table => Table): Table = bind(fn(this))
  def bind(fn: Table => Table): Table = bind(fn(this))
  def extend(other: Table): Table = bind(other)
  def extend(other: Column): Table = bind(other)

  // product
  def cross(other: Table)(implicit scope: Scope): Table = product(other)
  def product(other: Table)(implicit scope: Scope): Table = Scope {
    implicit scope =>
      val idx1 = STen.arange_l(
        0,
        this.numRows,
        1L,
        this.device.to(STen.lOptions)
      )
      val idx2 = STen.arange_l(
        0,
        other.numRows,
        1L,
        other.device.to(STen.lOptions)
      )
      val cartes = STen.cartesianProduct(List(idx1, idx2))
      val t1 = this.rows(cartes.select(1, 0))
      val t2 = other.rows(cartes.select(1, 1))
      t1.bind(t2)
  }
  def chunkedProduct(
      other: Table,
      chunkSize: Int = 5000
  ): Resource[IO, List[Resource[IO, Table]]] =
    Scope.inResource.map { implicit scope =>
      val idx1 = STen.arange_l(
        0,
        this.numRows,
        1L,
        this.device.to(STen.lOptions)
      )
      val idx2 = STen.arange_l(
        0,
        other.numRows,
        1L,
        other.device.to(STen.lOptions)
      )
      val cartes = STen.cartesianProduct(List(idx1, idx2))
      val l = cartes.shape(0)
      val b = chunkSize
      val ranges =
        0L until l / b map (i => (i * b, math.min(l, (i + 1) * b))) toList

      ranges.map { case (from, until) =>
        Scope.inResource.map { implicit scope =>
          val c = cartes.slice(0, from, until, 1)
          val t1 = this.rows(c.select(1, 0))
          val t2 = other.rows(c.select(1, 1))
          t1.bind(t2)
        }
      }

    }

  // distinct

  def distinct(implicit scope: Scope): Table =
    Scope { implicit scope =>
      val locations: IndexedSeq[STen] =
        this.groupBy(0 until numCols: _*).groups
      val aggregates = locations
        .map { locs =>
          Table(
            columns.map {
              _.select(locs.select(dim = 0, index = 0))
            },
            colNames
          )

        }

      Table.union(aggregates: _*)

    }

  // aggregation

  def groupBy[S: Sc](
      cols: ColumnSelection*
  ): TableWithGroups = {

    val (factorCounts, factors) = factorize(cols.map(resolveColumnIdx): _*)

    // this loop is quadratic
    // for large number of factors (groups) this should be done by sorting
    val groupLocations = 0L until factorCounts.shape(0) map { factor =>
      factors.values.equ(factor).where.head
    }

    TableWithGroups(this, groupLocations)

  }

  // selection
  def restrict(predicate: Column)(implicit
      scope: Scope
  ): Table = filter(predicate)
  def select(predicate: Column)(implicit
      scope: Scope
  ): Table = filter(predicate)

  def filter(p: Table => Column)(implicit scope: Scope): Table = select(p)
  def restrict(p: Table => Column)(implicit scope: Scope): Table = select(p)
  def select(p: Table => Column)(implicit scope: Scope): Table = {
    val pr = p(this)
    filter(pr)
  }
  def equifilter[A](p: TableExpression => EquExpression[A])(implicit scope: Scope):Table = {
    val EquExpression(column,value) = p(TableExpression(this))
    val indexed = column.indexed
    val locs = indexed.indexAs[A].get.index.get(value)
    this.rows(locs)
  }

  def filter(predicate: Column)(implicit
      scope: Scope
  ): Table =
    Scope { implicit scope =>
      val indices = predicate.values.where.head
      this.rows(indices)
    }

}

case class TableExpression(t:Table) {
  def apply(s:ColumnSelection) : ColumnExpression = ColumnExpression(t.colAt(t.resolveColumnIdx(s)))
}

case class ColumnExpression(c:Column) {
  def ===[T](a:T) = EquExpression(c,a)
  def eq[T](a:T) = EquExpression(c,a)
  def equ[T](a:T) = EquExpression(c,a)
}

case class EquExpression[A](column:Column, equalsWith: A)

case class TableWithGroups(table: Table, groups: IndexedSeq[STen]) {
  def colAt(idx: Int*): TableWithGroups =
    copy(table =
      Table(idx.map(i => table.columns(i)).toVector, table.colNames.at(idx: _*))
    )

  def cols(names: String*): TableWithGroups =
    copy(table =
      Table(
        table.colNames.apply(names: _*).map(i => table.columns(i)).toVector,
        table.colNames.at(table.colNames(names: _*))
      )
    )
  def project(name: String*): TableWithGroups = cols(name: _*)

  def withoutCol(s: Set[Int]): TableWithGroups = {
    colAt(table.columns.zipWithIndex.map(_._2).filterNot(s.contains): _*)
  }
  def withoutCol(s: Int): TableWithGroups = withoutCol(Set(s))
  def remove(s: Int*): TableWithGroups = withoutCol(s.toSet)
  def remove(s: Int): TableWithGroups = withoutCol(s)
  def remove(s: String): TableWithGroups = withoutCol(
    table.colNames.get(s).toSet
  )

  def transform[T](transform: STen => T): Vector[T] = {
    val builder = new scala.collection.immutable.VectorBuilder[T]
    groups.foreach { locs =>
      builder.addOne(transform(locs))
      ()
    }

    builder.result()
  }

  def aggregate(fun: Table => Table)(implicit scope: Scope): Table =
    Table.union(transform(locs => fun(table.rows(locs))): _*)

}

