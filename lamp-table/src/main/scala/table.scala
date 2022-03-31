package lamp.table

import lamp._
import org.saddle.scalar.ScalarTagLong
import org.saddle.scalar.ScalarTagFloat
import org.saddle.scalar.ScalarTagDouble
import org.saddle._
import org.saddle.order._
import org.saddle.Index
import org.saddle.index.OuterJoin
import lamp.saddle._
import org.saddle.index.IndexIntRange

object Table {

  implicit val movable: Movable[Table] = Movable.by(_.columns)

  def apply(cols: Column*): Table =
    Table(cols.toVector, IndexIntRange(cols.length).map(_.toString))

  def union[S: Sc](tables: Table*): Table =
    tables.head.union(tables.tail: _*)

}

case class Table(
    columns: Vector[Column],
    colNames: Index[String]
) extends RelationalAlgebra {
  require(columns.size == colNames.length)

  def withColNames(n: Index[String]): Table =
    Table(colNames = n, columns = columns)

  def withColumns(columns: Vector[Column]): Table = {
    require(columns.length == colNames.length)
    Table(columns = columns, colNames = colNames)
  }

  def colName(i: Int) = colNames.at(i).toOption

  def equalDeep(other: Table) = {
    val a1 = columns.map(v => (v.index, v.tpe))
    val a2 = other.columns.map(v => (v.index, v.tpe))
    a1 == a2 && colNames == other.colNames && {
      columns
        .map(_.values)
        .zip(other.columns.map(_.values))
        .map(v => v._1.equalDeep(v._2))
        .forall(identity)
    }
  }

  def toSTen[S: Sc] = {
    STen.cat(
      columns.map(_.values.castToType(7).view(numRows, -1)),
      dim = 1
    )
  }

  override def toString =
    stringify()

  def stringify(nrows: Int = 10, ncols: Int = 10) = Scope.unsafe {
    implicit scope =>
      val n = math.min(numRows, nrows).toInt
      val m = math.min(numCols, ncols)
      if (n == 0 || m == 0) "Empty Table"
      else {
        val columnIdxNeeded = (0 until m / 2) ++ (m / 2 until m)
        val rowIdxNeeded = ((0 until n / 2) ++ (n / 2 until n))

        val selected = colAt(columnIdxNeeded: _*).rows(rowIdxNeeded: _*)

        val stringFrame = selected.columns.zipWithIndex
          .map { case (column, idx) =>
            val name = colName(idx).get
            val indexed = if (column.index.isDefined) "|IX" else ""
            val frame = column.tpe match {
              case DateTimeColumnType(_) =>
                Frame(
                  s"$name|DT$indexed" -> column.values.toLongVec.map(l =>
                    if (ScalarTagLong.isMissing(l)) null
                    else java.time.Instant.ofEpochMilli(l).toString()
                  )
                )
              case BooleanColumnType(_) =>
                Frame(
                  s"$name|B$indexed" -> column.values.toLongVec.map(l =>
                    if (ScalarTagLong.isMissing(l)) null
                    else if (l == 0) "false"
                    else "true"
                  )
                )
              case TextColumnType(_, pad, vocabulary) =>
                val reverseVocabulary = vocabulary.map(_.map(_.swap))
                Frame(s"$name|TXT$indexed" -> column.values.toLongMat.rows.map {
                  row =>
                    if (row.count == 0) null
                    else
                      row
                        .filter(_ != pad)
                        .map(l =>
                          reverseVocabulary.map(_.apply(l)).getOrElse(l.toChar)
                        )
                        .toArray
                        .mkString
                }.toVec)
              case I64ColumnType =>
                val m =
                  column.values.view(-1, 1).toLongMat.map(ScalarTagLong.show)

                m.toFrame.setColIndex(
                  Index(0 until m.numCols map (_ => s"$name|I64$indexed"): _*)
                )
              case F32ColumnType =>
                val m =
                  column.values.view(-1, 1).toFloatMat.map(ScalarTagFloat.show)
                m.toFrame.setColIndex(
                  Index(0 until m.numCols map (_ => s"$name|F32$indexed"): _*)
                )
              case F64ColumnType =>
                val m =
                  column.values.view(-1, 1).toMat.map(ScalarTagDouble.show)
                m.toFrame.setColIndex(
                  Index(0 until m.numCols map (_ => s"$name|F64$indexed"): _*)
                )
            }
            frame
          }
          .reduce(_ rconcat _)
          .setRowIndex(Index(rowIdxNeeded: _*))

        stringFrame.stringify(nrows, ncols)
      }

  }

  def device: Device =
    columns.headOption.map(_.values.device).getOrElse(lamp.CPU)

  def copyToDevice[S: Sc](device: Device) =
    Table(
      columns.map { column => column.copy(column.values.copyToDevice(device)) },
      colNames
    )

  def numCols: Int = columns.length

  def numRows: Long = columns.headOption
    .map(_.values.shape.headOption.getOrElse(1L))
    .getOrElse(0L)

  def replaceCol(
      name: String,
      update: STen,
      tpe: Option[ColumnDataType]
  ): Table =
    replaceCol(colNames.getFirst(name), update, tpe)

  def replaceCol(
      idx: Int,
      update: STen,
      tpe: Option[ColumnDataType] = None
  ): Table = {
    val old = columns(idx)
    require(old.values.shape(0) == update.shape(0))
    Table(
      columns
        .updated(
          idx,
          Column(update, tpe.getOrElse(old.tpe), None)
        ),
      colNames
    )
  }
  def replaceCol(
      idx: Int,
      update: Column
  ): Table = {
    val old = columns(idx)
    require(old.values.shape(0) == update.values.shape(0))
    Table(
      columns
        .updated(
          idx,
          update
        ),
      colNames
    )
  }

  def mapCols(
      fun: (Column, String) => Column
  ): Table =
    withColumns(columns = columns.zip(colNames.toSeq).map(fun.tupled))

  def pivot(col0: Int, col1: Int)(
      selectAndAggregate: Table => Table
  )(implicit scope: Scope): Table = {
    val columns = this
      .groupBy[Table](col1)
      .transform { case samePivotLocs =>
        val samePivotTable = rows(samePivotLocs)
        val pivotValueAsString = samePivotTable
          .colAt(Seq(col1): _*)
          .rows(0)
          .columns
          .head
          .toVec(0)
          .toString
        samePivotTable.groupBy(col0).aggregate { case table =>
          table
            .colAt(Seq(col0): _*)
            .rows(0)
            .bind(
              selectAndAggregate(table)
                .rename(0, pivotValueAsString)
            )
        }
      }
      .toSeq
    columns.reduceLeft((a, b) => a.equijoin(0, b, 0, OuterJoin))
  }

  def indexed(colIdx: Seq[Int] = Nil, names: Seq[String] = Nil): Table = {

    val toUpdate =
      (colIdx ++ names.flatMap(n => colNames(n).toList)).distinct
    val updated = toUpdate.foldLeft(columns)((columns, idx) =>
      columns.updated(idx, columns(idx).withIndex)
    )
    withColumns(columns = updated)

  }

  def colType(idx: Int): ColumnDataType = columns(idx).tpe
  def colType(name: String): ColumnDataType = columns(
    colNames.getFirst(name)
  ).tpe

  def take[S: Sc](idx: STen): Table = rows(idx)
  def rows[S: Sc](idx: STen): Table = {
    withColumns(
      columns = columns.map { column =>
        column.select(idx)
      }
    )
  }

  def rows(idx: Int*)(implicit scope: Scope): Table = rows(idx.toArray)
  def take(idx: Int*)(implicit scope: Scope): Table = rows(idx.toArray)
  def take(idx: Array[Int])(implicit scope: Scope): Table = rows(idx)

  def rows(idx: Array[Int])(implicit scope: Scope): Table = {
    import org.saddle._
    val vidx = idx.toVec.map(_.toLong)
    if (vidx.countif(_ < 0) == 0)
      rows(lamp.saddle.fromLongVec(vidx, device = device))
    else {
      withColumns(columns = columns.map { case Column(sten, tpe, _) =>
        Scope { implicit scope =>
          val shape = vidx.length.toLong :: sten.shape.drop(1)
          val missing = STen.zeros(shape, sten.options.toDouble)
          missing.fill_(Double.NaN)
          val cast = missing.castToType(sten.scalarTypeByte)

          val nonmissingIdxLocationV = vidx.find(_ >= 0L)
          val nonmissingIdxLocation =
            lamp.saddle
              .fromLongVec(nonmissingIdxLocationV.map(_.toLong), device)
              .view(
                nonmissingIdxLocationV.length.toLong :: sten.shape
                  .drop(1)
                  .map(_ => 1L): _*
              )
              .expand(
                nonmissingIdxLocationV.length.toLong :: sten.shape.drop(1)
              )
          val nonmissingIdxValueV = vidx.take(nonmissingIdxLocationV.toArray)
          val nonmissingIdxValue =
            lamp.saddle.fromLongVec(nonmissingIdxValueV.map(_.toLong), device)

          val nonmissingValues = sten.indexSelect(0, nonmissingIdxValue)

          val ret = cast.scatter(0, nonmissingIdxLocation, nonmissingValues)
          Column(ret, tpe, None)
        }

      })

    }
  }

  def factorize[S: Sc](cols: Int*): (STen, Column) = {
    val stacked =
      if (cols.size == 1) columns(cols.head).values
      else {
        val factorizedColumns = cols.map { col =>
          val values = columns(col).values
          val (uniques, uniqueLocations, _) = values.unique(
            dim = 0,
            sorted = false,
            returnInverse = true,
            returnCounts = false
          )
          val uniqueIds =
            STen.arange_l(0, uniques.shape(0), 1, uniqueLocations.options)
          uniqueIds.indexSelect(0, uniqueLocations)
        }
        STen.stack(factorizedColumns, 1)
      }
    val (uniques, uniqueLocations, counts) = stacked.unique(
      dim = 0,
      sorted = false,
      returnInverse = true,
      returnCounts = true
    )

    val uniqueIds =
      STen.arange_l(0, uniques.shape(0), 1, uniqueLocations.options)
    (counts, Column(uniqueIds.indexSelect(0, uniqueLocations)))

  }

  def rfilter(
      predicateOnColumns: ColumnSelection*
  )(predicate: Series[String, _] => Boolean)(implicit scope: Scope): Table = {
    val proj = {
      if (predicateOnColumns.isEmpty) this
      else {
        val columnIds = predicateOnColumns.map(this.resolveColumnIdx)
        this.colAt(columnIds: _*)
      }
    }
    val vecs = proj.columns.map(_.toVec)
    val buffer = org.saddle.Buffer.empty[Int]
    var i = 0L
    val N = proj.numRows
    while (i < N) {
      if (predicate(Series(proj.colNames, vecs.map(_.raw(i.toInt)).toVec))) {
        buffer.+=(i.toInt)
      }
      i += 1
    }
    this.rows(buffer.toArray)
  }

}
