package lamp.tgnn

import lamp._
import java.nio.channels.ReadableByteChannel
import lamp.io.csv.asciiSilentCharsetDecoder
import java.nio.charset.CharsetDecoder
import lamp.io.csv.Buffer
import org.saddle.scalar.ScalarTagLong
import org.saddle.scalar.ScalarTagFloat
import org.saddle.scalar.ScalarTagDouble
import java.io.File
import org.saddle.Index
import org.saddle.index.InnerJoin

case class Table(
    columns: Vector[(STen, Option[String])],
    indices: Map[Int, Index[Long]]
) {

  def fuse[S: Sc] = {
    val highestTpe = columns.map(_._1.scalarTypeByte).max
    STen.cat(
      columns.map(_._1.castToType(highestTpe).view(numRows, -1)),
      dim = 1
    )
  }

  override def toString =
    s"Table(\n[$numRows x $numCols]\n\tName\tShape\tType\tTensor\n${columns.zipWithIndex
      .map { case ((sten, name), idx) =>
        idx.toString + ".\t" + name.getOrElse("_") + "\t" + sten.shape.mkString("[", ",", "]") + "\t" + sten.scalarTypeByte + "\t" + sten
      }
      .mkString("\n")}\n)"

  def stringify(nrows: Int = 10, ncols: Int = 10) = Scope.leak {
    implicit scope =>
      val n = math.min(numRows, nrows).toInt
      val m = math.min(numCols, ncols)
      val a = cols(0 until m / 2: _*).rows((0 until n / 2).toArray).fuse
      val b =
        cols(m / 2 until m: _*).rows((0 until n / 2).toArray).fuse
      val c =
        cols(0 until m / 2: _*).rows((n / 2 until n).toArray).fuse
      val d = cols((m / 2 until m): _*)
        .rows((n / 2 until n).toArray)
        .fuse

      def rep(i: Int, s: String) = {
        val sh = col(i).shape
        val repeats = sh match {
          case List(_)    => 1
          case List(_, x) => x.toInt
          case _          => ???
        }
        (0 until repeats) map (_ => s)
      }

      val colnames =
        (0 until m / 2).flatMap(i => rep(i, colName(i).getOrElse("_" + i))) ++
          (m / 2 until m).flatMap(i => rep(i, colName(i).getOrElse("_" + i)))

      ((a
        .cat(b, dim = 1))
        .cat((c.cat(d, dim = 1)), dim = 0))
        .toMat
        .toFrame
        .setColIndex(Index(colnames: _*))
        .stringify(nrows, ncols)

  }

  def device: Device = columns.headOption.map(_._1.device).getOrElse(lamp.CPU)

  def copyToDevice[S: Sc](device: Device) =
    Table(columns.map { case (st, n) => (st.copyToDevice(device), n) }, indices)

  def numCols: Int = columns.length

  def numRows: Long = columns.headOption.map(_._1.shape(0)).getOrElse(0L)

  def colNames: Vector[Option[String]] = columns.map(_._2)

  def groupBy[S: Sc](col: Int)(transform: Table => Table): Table = {
    val index = indexCols(List(col)).indices(col)

    val uniq = index.uniques.toVec

    val tables = uniq.map { key =>
      val locs = index.get(key)
      val row = rows(locs)
      transform(row)
    }

    tables(0).union(tables.toSeq.tail: _*)

  }

  val nameToIndex: Map[String, Int] = columns.zipWithIndex
    .map(v => v._1._2 -> v._2)
    .collect { case (Some(value), idx) => (value, idx) }
    .toMap

  def indexCols(colIdx: Seq[Int] = Nil, names: Seq[String] = Nil): Table = {
    copy(indices =
      indices ++ ((colIdx ++ names.flatMap(n => nameToIndex.get(n).toList))
        .filter(i =>
          !indices
            .contains(i)
        )
        .distinct
        .map { idx =>
          idx -> Index(columns(idx)._1.toLongVec)
        })
    )
  }

  def col(idx: Int): STen = columns(idx)._1

  def col(name: String): STen = col(nameToIndex(name))

  def colName(idx: Int): Option[String] = columns(idx)._2

  def cols(idx: Int*): Table = {
    val map = idx.zipWithIndex.toMap
    val c = idx.map(i => columns(i)).toVector
    val i = indices.map { case (oldIdx, index) => (map(oldIdx), index) }
    Table(c, i)
  }

  def withoutCol(s: Set[Int]) = {
    cols(columns.zipWithIndex.map(_._2).filterNot(s.contains): _*)
  }

  def updateCol(name: String, update: STen): Table =
    updateCol(nameToIndex(name), update)

  def updateCol(idx: Int, update: STen): Table = {
    val old = columns(idx)
    require(old._1.shape(0) == update.shape(0))
    Table(
      columns.updated(idx, (update, old._2)),
      indices = indices.removed(idx)
    )
  }

  def mapColNames(fun: (Option[String], Int) => Option[String]) = Table(
    columns.map(_._1).zip(columns.map(_._2).zipWithIndex.map(fun.tupled)),
    indices
  )

  def mapCols(
      fun: STen => STen
  ): Table = Table(columns.map(_._1).map(fun).zip(columns.map(_._2)))

  def join[S: Sc](col: Int, other: Table, otherCol: Int): Table = {
    val indexA = indexCols(List(col)).indices(col)
    val indexB = other.indexCols(List(otherCol)).indices(otherCol)
    val reindexer = indexA.join(indexB, InnerJoin)
    val asub = reindexer.lTake.map(i => this.rows(i)).getOrElse(this)
    val bsub = reindexer.rTake
      .map(i => other.withoutCol(Set(otherCol)).rows(i))
      .getOrElse(other.withoutCol(Set(otherCol)))
    asub.bind(bsub)
  }

  def union[S: Sc](others: Table*): Table = {
    val c = (0 until numCols).map { colIdx =>
      val name = colName(colIdx)
      val s1 = col(colIdx)
      val s3 = STen.cat(List(s1) ++ others.map(_.col(colIdx)), dim = 0)
      (s3, name)
    }.toVector
    Table(c)
  }

  def bind(other: Table): Table = {
    require(numRows == other.numRows)
    val c = columns ++ other.columns
    val i = (other.indices.map { case (oldIdx, index) =>
      (oldIdx + this.numCols) -> index
    }) ++ this.indices
    Table(c, i)
  }

  def bind(col: STen): Table = bind(Table(Vector((col, None))))

  def rows[S: Sc](idx: STen): Table = {
    Table(
      columns.map { case (sten, name) =>
        (sten.indexSelect(dim = 0, index = idx), name)
      }
    )
  }

  def rows[S: Sc](idx: Array[Int]): Table = {
    import org.saddle._
    rows(STen.fromLongVec(idx.toVec.map(_.toLong), device = device))
  }

}

object Table {

  def unnamed(cols: STen*): Table =
    Table(cols.map((_, None)).toVector, Map.empty[Int, Index[Long]])
    
  def apply(cols: Seq[(STen, Option[String])]): Table =
    Table(cols.toVector, Map.empty[Int, Index[Long]])

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
          (ondevice, name)
      }
      if (columns.map(_._1.shape(0)).distinct.size != 1)
        Left(s"Uneven length ${columns.map(_._1.shape(0)).toVector} columns")
      else {

        Right(Table(columns.toVector))
      }
    }

  }


}
