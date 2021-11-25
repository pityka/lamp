package lamp.tgnn

import lamp._
import java.nio.channels.ReadableByteChannel
import lamp.io.csv.asciiSilentCharsetDecoder
import java.nio.charset.CharsetDecoder
import lamp.io.csv.Buffer
import org.saddle.scalar.ScalarTagLong
import org.saddle.scalar.ScalarTagFloat
import org.saddle.scalar.ScalarTagDouble
import org.saddle._
import org.saddle.order._
import java.io.File
import org.saddle.Index
import org.saddle.index.InnerJoin
import lamp.tgnn.Table.DateTimeColumn
import lamp.tgnn.Table.BooleanColumn
import lamp.tgnn.Table.TextColumn
import lamp.tgnn.Table.I64Column
import lamp.tgnn.Table.F32Column
import lamp.tgnn.Table.F64Column
import org.saddle.index.OuterJoin

case class Table(
    columns: Vector[Table.Column[_]]
) {

  def equalDeep(other: Table) = {
    val a1 = columns.map(v => (v.index, v.name, v.tpe))
    val a2 = other.columns.map(v => (v.index, v.name, v.tpe))
    a1 == a2 && {
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
    s"Table(\n[$numRows x $numCols]\n\tName\tShape\tValueType\tType\tTensor\tIndeed\n${columns.zipWithIndex
      .map { case (Table.Column(sten, name, tpe, index), idx) =>
        idx.toString + ".\t" + name
          .getOrElse("_") + "\t" + sten.shape.mkString("[", ",", "]") + "\t" + sten.scalarTypeByte + "\t" + tpe + "\t" + sten + "\t" + index.isDefined
      }
      .mkString("\n")}\n)"

  def stringify(nrows: Int = 10, ncols: Int = 10) = Scope.leak {
    implicit scope =>
      val n = math.min(numRows, nrows).toInt
      val m = math.min(numCols, ncols)
      if (n == 0 || m == 0) "Empty Table"
      else {
        val columnIdxNeeded = (0 until m / 2) ++ (m / 2 until m)
        val rowIdxNeeded = (0 until n / 2) ++ (n / 2 until n)

        val selected = cols(columnIdxNeeded: _*).rows(rowIdxNeeded: _*)

        val stringFrame = selected.columns
          .map { column =>
            val name = column.name.getOrElse("")
            val frame = column.tpe match {
              case DateTimeColumn(_) =>
                Frame(
                  name -> column.values.toLongVec.map(l =>
                    if (ScalarTagLong.isMissing(l)) null
                    else java.time.Instant.ofEpochMilli(l).toString()
                  )
                )
              case BooleanColumn(_) =>
                Frame(
                  name -> column.values.toLongVec.map(l =>
                    if (ScalarTagLong.isMissing(l)) null
                    else if (l == 0) "false"
                    else "true"
                  )
                )
              case TextColumn(_, pad, vocabulary) =>
                val reverseVocabulary = vocabulary.map(_.map(_.swap))
                Frame(name -> column.values.toLongMat.rows.map { row =>
                  if (row.countif(ScalarTagLong.isMissing) > 0) null
                  else
                    row
                      .filter(_ != pad)
                      .map(l =>
                        reverseVocabulary.map(_.apply(l)).getOrElse(l.toChar)
                      )
                      .toArray
                      .mkString
                }.toVec)
              case I64Column =>
                val m = column.values.toLongMat.map(ScalarTagLong.show)

                m.toFrame.setColIndex(
                  Index(0 until m.numCols map (_ => name): _*)
                )
              case F32Column =>
                val m = column.values.toFloatMat.map(ScalarTagFloat.show)
                m.toFrame.setColIndex(
                  Index(0 until m.numCols map (_ => name): _*)
                )
              case F64Column =>
                val m = column.values.toMat.map(ScalarTagDouble.show)
                m.toFrame.setColIndex(
                  Index(0 until m.numCols map (_ => name): _*)
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
      columns.map { column => column.copy(column.values.copyToDevice(device)) }
    )

  def numCols: Int = columns.length

  def numRows: Long = columns.headOption.map(_.values.shape.headOption.getOrElse(1L)).getOrElse(0L)

  def colNames: Vector[Option[String]] = columns.map(_.name)

  def groupByGroupIndices[S: Sc](
      cols: Int*
  ): IndexedSeq[STen] = {

    val factorized = cols.map { col =>
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

    val cartesianProductOfUniqueIds = STen.cartesianProduct(factorized.toList)
    val (groups, groupLocations, _) = cartesianProductOfUniqueIds.unique(
      dim = 0,
      sorted = false,
      returnInverse = true,
      returnCounts = false
    )

    // this loop is quadratic
    // this should be done by sorting
    0L until groups.shape(0) map (groupId =>
      groupLocations.equ(groupId).where.head
    )

  }

  def groupBy[T](
      cols: Int*
  )(transform: STen => T)(implicit scope: Scope): Vector[T] = {
    val indices = groupByGroupIndices(cols: _*)
    val builder = new scala.collection.immutable.VectorBuilder[T]
    indices.foreach { locs =>
      builder.addOne(transform(locs))
      ()
    }

    builder.result()

  }

  def groupByThenUnion(
      col: Int
  )(transform: STen => Table)(implicit scope: Scope): Table = {
    val tables = groupBy[Table](col)(transform)
    tables(0).union(tables.toSeq.tail: _*)
  }

  def pivot(col0: Int, col1: Int)(
      selectAndAggregate: Table => Table
  )(implicit scope: Scope): Table = {
    val columns = this
      .groupBy[Table](col1) { case samePivotLocs =>
        val samePivotTable = rows(samePivotLocs)
        val pivotValueAsString = samePivotTable
          .cols(col1)
          .rows(0)
          .columns
          .head
          .toVec(0)
          .toString
        samePivotTable.groupByThenUnion(col0) { case samePivotSameKeyLocs =>
          val table = rows(samePivotSameKeyLocs)

          table
            .cols(col0)
            .rows(0)
            .bind(
              selectAndAggregate(table)
                .updateColName(0, Some(pivotValueAsString))
            )
        }
      }
      .toSeq
    columns.reduceLeft((a, b) => a.join(0, b, 0, OuterJoin))
  }

  val nameToIndex: Map[String, Int] = columns.zipWithIndex
    .map(v => v._1.name -> v._2)
    .collect { case (Some(value), idx) => (value, idx) }
    .toMap

  def indexCols(colIdx: Seq[Int] = Nil, names: Seq[String] = Nil): Table = {

    val toUpdate =
      (colIdx ++ names.flatMap(n => nameToIndex.get(n).toList)).distinct
    val updated = toUpdate.foldLeft(columns)((columns, idx) =>
      columns.updated(idx, columns(idx).withIndex)
    )
    copy(columns = updated)

  }

  def col(idx: Int): STen = columns(idx).values

  def col(name: String): STen = col(nameToIndex(name))

  def colName(idx: Int): Option[String] = columns(idx).name

  def colType(idx: Int): Table.ColumnDataType = columns(idx).tpe
  def colType(name: String): Table.ColumnDataType = columns(
    nameToIndex(name)
  ).tpe

  def cols(idx: Int*): Table =
    Table(idx.map(i => columns(i)).toVector)

  def withoutCol(s: Int): Table = withoutCol(Set(s))
  def withoutCol(s: Set[Int]) = {
    cols(columns.zipWithIndex.map(_._2).filterNot(s.contains): _*)
  }
  def updateColName(name: String, newName: Option[String]): Table =
    updateColName(nameToIndex(name), newName)
  def updateColName(idx: Int, newName: Option[String]): Table =
    Table(columns.updated(idx, columns(idx).copy(name = newName)))
  def updateColType(name: String, newTpe: Table.ColumnDataType): Table =
    updateColType(nameToIndex(name), newTpe)
  def updateColType(idx: Int, newTpe: Table.ColumnDataType): Table =
    Table(columns.updated(idx, columns(idx).copy(tpe = newTpe)))

  def updateCol(
      name: String,
      update: STen,
      tpe: Option[Table.ColumnDataType]
  ): Table =
    updateCol(nameToIndex(name), update, tpe)

  def updateCol(
      idx: Int,
      update: STen,
      tpe: Option[Table.ColumnDataType] = None
  ): Table = {
    val old = columns(idx)
    require(old.values.shape(0) == update.shape(0))
    Table(
      columns
        .updated(
          idx,
          Table.Column(update, old.name, tpe.getOrElse(old.tpe), None)
        )
    )
  }

  def mapColNames(fun: (Option[String], Int) => Option[String]) = Table(
    columns.zip(columns.map(_.name).zipWithIndex.map(fun.tupled)).map {
      case (old, newname) => old.copy(name = newname)
    }
  )

  def mapCols(
      fun: STen => STen
  ): Table = Table(columns.map(_.values).map(fun).zip(columns).map {
    case (v, old) => old.copy(values = v)
  })

  def join[IndexType](
      col: Int,
      other: Table,
      otherCol: Int,
      how: org.saddle.index.JoinType = InnerJoin
  )(implicit scope: Scope): Table = {
    val indexA = indexCols(List(col)).columns(col).indexAs[IndexType].get.index
    val indexB = other
      .indexCols(List(otherCol))
      .columns(otherCol)
      .indexAs[IndexType]
      .get
      .index
    val reindexer = indexA.join(indexB, how)

    val asub = reindexer.lTake.map(i => this.rows(i)).getOrElse(this)
    val bsub = reindexer.rTake
      .map(i => other.rows(i))
      .getOrElse(other)

    val a =
      if (how == org.saddle.index.RightJoin)
        asub.withoutCol(Set(col))
      else asub
    val b =
      if (
        how == org.saddle.index.LeftJoin || how == org.saddle.index.InnerJoin || how == org.saddle.index.OuterJoin
      )
        bsub.withoutCol(Set(otherCol))
      else bsub

    val bind = a.bind(b)

    if (how == OuterJoin) {

      val kA = asub.columns(col).values
      val kB = bsub.columns(otherCol).values

      val idx =
        reindexer.rTake.getOrElse(array.range(0, other.numRows.toInt)).toVec

      val nonmissingIdxLocationV = idx.find(_ >= 0L)
      val nonmissingIdxLocation =
        STen
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
        STen.fromLongVec(nonmissingIdxValueV.map(_.toLong), device)

      val nonmissingValues =
        other.col(otherCol).indexSelect(0, nonmissingIdxValue)

      val ret = kA.scatter(0, nonmissingIdxLocation, nonmissingValues)
      val merged = asub.columns(col).copy(values = ret)
      Table(bind.columns.updated(col, merged))

    } else bind

  }

  def union[S: Sc](others: Table*): Table = {
    val c = (0 until numCols).map { colIdx =>
      val name = colName(colIdx)
      val tpe = colType(colIdx)
      val s1 = col(colIdx)
      val s3 = STen.cat(List(s1) ++ others.map(_.col(colIdx)), dim = 0)
      Table.Column(s3, name, tpe, None)
    }.toVector
    Table(c)
  }

  def bind(other: Table): Table = {
    require(numRows == other.numRows)
    val c = columns ++ other.columns
    Table(c)
  }

  def bind(col: STen): Table = bind(Table.unnamed(col))

  def rows[S: Sc](idx: STen): Table = {
    Table(
      columns.map { column =>
        column.select(idx)
      }
    )
  }

  def rows(idx: Int*)(implicit scope: Scope): Table = rows(idx.toArray)

  def rows(idx: Array[Int])(implicit scope: Scope): Table = {
    import org.saddle._
    val vidx = idx.toVec.map(_.toLong)
    if (vidx.countif(_ < 0) == 0)
      rows(STen.fromLongVec(vidx, device = device))
    else {
      Table(columns.map { case Table.Column(sten, name, tpe, _) =>
        Scope { implicit scope =>
          val shape = vidx.length.toLong :: sten.shape.drop(1)
          val missing = STen.zeros(shape, sten.options.toDouble)
          missing.fill_(Double.NaN)
          val cast = missing.castToType(sten.scalarTypeByte)

          val nonmissingIdxLocationV = vidx.find(_ >= 0L)
          val nonmissingIdxLocation =
            STen
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
            STen.fromLongVec(nonmissingIdxValueV.map(_.toLong), device)

          val nonmissingValues = sten.indexSelect(0, nonmissingIdxValue)

          val ret = cast.scatter(0, nonmissingIdxLocation, nonmissingValues)
          Table.Column(ret, name, tpe, None)
        }

      })

    }
  }

}

object Table {

  implicit val movable: Movable[Table] = Movable.by(_.columns)

  def dataTypeFromScalarTypeByte(s: Byte) = s match {
    case 7 => F64Column
    case 6 => F32Column
    case 5 => I64Column
  }

  def unnamed(cols: STen*): Table =
    Table(
      cols
        .map(s =>
          Table
            .Column(s, None, dataTypeFromScalarTypeByte(s.scalarTypeByte), None)
        )
        .toVector
    )

  sealed trait ColumnIndex[T] {
    def index: Index[T]
    implicit def st: ST[T]
    implicit def ord: ORD[T]
    def uniqueLocations: Array[Vec[Int]] = {
      val u = index.uniques
      u.toVec.toArray.map { l =>
        index.get(l).toVec
      }
    }

  }
  case class LongIndex(index: Index[Long]) extends ColumnIndex[Long] {
    val st = ScalarTagLong
    val ord = implicitly[ORD[Long]]
  }
  case class FloatIndex(index: Index[Float]) extends ColumnIndex[Float] {
    val st = ScalarTagFloat
    val ord = implicitly[ORD[Float]]
  }
  case class DoubleIndex(index: Index[Double]) extends ColumnIndex[Double] {
    val st = ScalarTagDouble
    val ord = implicitly[ORD[Double]]
  }
  case class StringIndex(index: Index[String]) extends ColumnIndex[String] {
    val st = implicitly[ST[String]]
    val ord = implicitly[ORD[String]]
  }

  case class Column[T](
      values: STen,
      name: Option[String],
      tpe: Table.ColumnDataType,
      index: Option[ColumnIndex[_]]
  ) {

    def select[S: Sc](idx: STen): Column[T] =
      Table.Column(values.indexSelect(dim = 0, index = idx), name, tpe, None)

    def indexAs[A] = index.asInstanceOf[Option[ColumnIndex[A]]]

    def withIndex = if (index.isDefined) this else copy(index = Some(makeIndex))

    def withName(s:String) = copy(name = Some(s))
    def withName(s:Option[String]) = copy(name = s)

    def toVec: Vec[_] = tpe match {
      case DateTimeColumn(_) =>
        values.toLongVec

      case BooleanColumn(_) =>
        values.toLongVec

      case TextColumn(_, pad, vocabulary) =>
        val reverseVocabulary = vocabulary.map(_.map(_.swap))
        val vec = values.toLongMat.rows.map { row =>
          if (row.countif(ScalarTagLong.isMissing) > 0) null
          else
            row
              .filter(_ != pad)
              .map(l => reverseVocabulary.map(_.apply(l)).getOrElse(l.toChar))
              .toArray
              .mkString
        }.toVec
        vec
      case I64Column =>
        val m = Scope.leak { implicit scope =>
          values.view(values.shape(0), -1).select(1, 0).toLongVec
        }
        m
      case F32Column =>
        val m = Scope.leak { implicit scope =>
          values.view(values.shape(0), -1).select(1, 0).toFloatVec
        }
        m
      case F64Column =>
        val m = Scope.leak { implicit scope =>
          values.view(values.shape(0), -1).select(1, 0).toVec
        }
        m
    }
    def makeIndex: ColumnIndex[_] = tpe match {
      case DateTimeColumn(_) =>
        LongIndex(Index(values.toLongVec))

      case BooleanColumn(_) =>
        LongIndex(Index(values.toLongVec))

      case TextColumn(_, pad, vocabulary) =>
        val reverseVocabulary = vocabulary.map(_.map(_.swap))
        val vec = values.toLongMat.rows.map { row =>
          if (row.countif(ScalarTagLong.isMissing) > 0) null
          else
            row
              .filter(_ != pad)
              .map(l => reverseVocabulary.map(_.apply(l)).getOrElse(l.toChar))
              .toArray
              .mkString
        }.toVec
        StringIndex(Index(vec))
      case I64Column =>
        val m = Scope.leak { implicit scope =>
          values.view(values.shape(0), -1).select(1, 0).toLongVec
        }
        LongIndex(Index(m))
      case F32Column =>
        val m = Scope.leak { implicit scope =>
          values.view(values.shape(0), -1).select(1, 0).toFloatVec
        }
        FloatIndex(Index(m))
      case F64Column =>
        val m = Scope.leak { implicit scope =>
          values.view(values.shape(0), -1).select(1, 0).toVec
        }
        DoubleIndex(Index(m))
    }
  }
  object Column {
    implicit val movable: Movable[Column[_]] = Movable.by(_.values)
    def bool(s:STen) = Column(s,None,BooleanColumn(),None)
  }

  sealed trait ColumnDataType {
    type Buf
    def allocateBuffer(): Buf
    def parseIntoBuffer(string: String, buffer: Buf): Unit
    def copyBufferToSTen(buf: Buf)(implicit scope: Scope): STen

  }
  final case class DateTimeColumn(
      parse: String => Long = DateTimeColumn.parse _
  ) extends ColumnDataType {
    type Buf = Buffer[Long]
    def allocateBuffer() = Buffer.empty[Long](1024)
    def parseIntoBuffer(string: String, buffer: Buf): Unit =
      buffer.+=(parse(string))

    def copyBufferToSTen(buf: Buffer[Long])(implicit scope: Scope): STen =
      STen.cat(
        buf.toArrays.map { ar =>
          STen
            .fromLongArray(
              ar,
              List(ar.length),
              CPU
            )
        },
        dim = 0
      )
  }
  object DateTimeColumn {
    def parse(s: String): Long = java.time.Instant.parse(s).toEpochMilli()
  }
  final case class BooleanColumn(
      isTrue: String => Boolean = BooleanColumn.parse _
  ) extends ColumnDataType {
    type Buf = Buffer[Long]
    def allocateBuffer() = Buffer.empty[Long](1024)
    def parseIntoBuffer(string: String, buffer: Buf): Unit =
      buffer.+=(if (isTrue(string)) 1L else 0L)

    def copyBufferToSTen(buf: Buffer[Long])(implicit scope: Scope): STen =
      STen.cat(
        buf.toArrays.map { ar =>
          STen
            .fromLongArray(
              ar,
              List(ar.length),
              CPU
            )
        },
        dim = 0
      )
  }
  object BooleanColumn {
    def parse(s: String) =
      s == "true" || s == "T" || s == "True" || s == "TRUE" || s == "yes" || s == "Yes" || s == "Yes" || s.trim
        .toLowerCase() == "yes" || s.trim.toLowerCase() == "true"
  }
  final case class TextColumn(
      maxLength: Int,
      pad: Long,
      vocabulary: Option[Map[Char, Long]]
  ) extends ColumnDataType {

    def tokenize(string: String): Array[Long] = string
      .map(char => vocabulary.map(_.apply(char)).getOrElse(char.toLong))
      .toArray
      .take(maxLength)
      .padTo(maxLength, pad)

    /* (tokens, mask) */
    type Buf = Buffer[Array[Long]]
    def allocateBuffer() = Buffer.empty[Array[Long]](1024)
    def parseIntoBuffer(string: String, buffer: Buf): Unit =
      buffer.+=(tokenize(string))

    def copyBufferToSTen(
        buf: Buffer[Array[Long]]
    )(implicit scope: Scope): STen =
      STen.cat(
        buf.toArrays.map { ar =>
          STen
            .fromLongArrayOfArrays(
              ar,
              List(ar.length, maxLength),
              CPU
            )
        },
        dim = 0
      )
  }
  case object I64Column extends ColumnDataType {
    type Buf = Buffer[Long]
    def allocateBuffer() = Buffer.empty[Long](1024)
    def parseIntoBuffer(string: String, buffer: Buf): Unit =
      buffer.+=(ScalarTagLong.parse(string))

    def copyBufferToSTen(buf: Buffer[Long])(implicit scope: Scope): STen =
      STen.cat(
        buf.toArrays.map { ar =>
          STen
            .fromLongArray(
              ar,
              List(ar.length),
              CPU
            )
        },
        dim = 0
      )
  }
  case object F32Column extends ColumnDataType {
    type Buf = Buffer[Float]
    def allocateBuffer() = Buffer.empty[Float](1024)
    def parseIntoBuffer(string: String, buffer: Buf): Unit =
      buffer.+=(ScalarTagFloat.parse(string))

    def copyBufferToSTen(buf: Buffer[Float])(implicit scope: Scope): STen =
      STen.cat(
        buf.toArrays.map { ar =>
          STen
            .fromFloatArray(
              ar,
              List(ar.length),
              CPU
            )
        },
        dim = 0
      )
  }
  case object F64Column extends ColumnDataType {
    type Buf = Buffer[Double]
    def allocateBuffer() = Buffer.empty[Double](1024)
    def parseIntoBuffer(string: String, buffer: Buf): Unit =
      buffer.+=(ScalarTagDouble.parse(string))

    def copyBufferToSTen(buf: Buffer[Double])(implicit scope: Scope): STen =
      STen.cat(
        buf.toArrays.map { ar =>
          STen
            .fromDoubleArray(
              ar,
              List(ar.length),
              CPU,
              DoublePrecision
            )
        },
        dim = 0
      )
  }

  def readHeterogeneousFromCSVFile(
      columnTypes: Seq[(Int, ColumnDataType)],
      file: File,
      device: Device = lamp.CPU,
      charset: CharsetDecoder = asciiSilentCharsetDecoder,
      fieldSeparator: Char = ',',
      quoteChar: Char = '"',
      recordSeparator: String = "\r\n",
      maxLines: Long = Long.MaxValue,
      header: Boolean = false
  )(implicit
      scope: Scope
  ): Either[String, Table] = {
    val fis = new java.io.FileInputStream(file)
    val channel = fis.getChannel
    try {
      readHeterogeneousFromCSVChannel(
        columnTypes,
        channel,
        device,
        charset,
        fieldSeparator,
        quoteChar,
        recordSeparator,
        maxLines,
        header
      )
    } finally {
      fis.close
    }
  }
  def readHeterogeneousFromCSVChannel(
      columnTypes: Seq[(Int, ColumnDataType)],
      channel: ReadableByteChannel,
      device: Device = lamp.CPU,
      charset: CharsetDecoder = asciiSilentCharsetDecoder,
      fieldSeparator: Char = ',',
      quoteChar: Char = '"',
      recordSeparator: String = "\r\n",
      maxLines: Long = Long.MaxValue,
      header: Boolean = false
  )(implicit
      scope: Scope
  ): Either[String, Table] = {

    val source = org.saddle.io.csv
      .readChannel(channel, bufferSize = 65536, charset = charset)

    val sortedColumnTypes = columnTypes
      .sortBy(_._1)
      .toIndexedSeq

    var bufdata: Seq[_] = null

    def prepare(headerLength: Int) = {
      val _ = headerLength
      bufdata = sortedColumnTypes.map { case (_, tpe) =>
        tpe.allocateBuffer()
      }
    }

    def addToBuffer(s: String, buf: Int) = {
      val tpe = sortedColumnTypes(buf)._2
      tpe.parseIntoBuffer(s, bufdata(buf).asInstanceOf[tpe.Buf])
    }

    val done = org.saddle.io.csv.parseFromIteratorCallback(
      source,
      sortedColumnTypes.map(_._1),
      fieldSeparator,
      quoteChar,
      recordSeparator,
      maxLines,
      header
    )(prepare, addToBuffer)

    done.flatMap { colIndex =>
      assert(bufdata.length == sortedColumnTypes.length)
      val columns = bufdata.zip(sortedColumnTypes).zipWithIndex map {
        case ((b, (_, tpe)), idx) =>
          val sten = tpe.copyBufferToSTen(b.asInstanceOf[tpe.Buf])
          val ondevice = if (device != CPU) {
            device.to(sten)
          } else sten
          val name = colIndex.map(_.apply(idx))
          Table.Column(ondevice, name, tpe, None)
      }
      if (columns.map(_.values.shape(0)).distinct.size != 1)
        Left(
          s"Uneven length ${columns.map(_.values.shape(0)).toVector} columns"
        )
      else {

        Right(Table(columns.toVector))
      }
    }

  }

    def union[S: Sc](tables: Table*): Table = 
      tables.head.union(tables.tail:_*)
  

}
