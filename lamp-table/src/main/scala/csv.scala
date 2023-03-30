package lamp.table

import lamp._
import java.nio.channels.ReadableByteChannel
import lamp.io.csv.makeAsciiSilentCharsetDecoder
import java.nio.charset.CharsetDecoder
import java.io.File
import java.nio.channels.WritableByteChannel
import java.nio.charset.Charset
import java.nio.charset.StandardCharsets
import java.nio.ByteBuffer
import org.saddle.Buffer
import org.saddle.io.csv.Callback

object csv {
  def renderToCSVString(
      table: Table,
      batchSize: Int = 10000,
      columnSeparator: Char = ',',
      quoteChar: Char = '"',
      recordSeparator: String = "\r\n",
      charset: Charset = StandardCharsets.UTF_8
  ) = {
    val buffer = Buffer.empty[Byte]
    val ch = new WritableByteChannel {
      def close = ()
      def isOpen = true
      def write(bb: ByteBuffer) = {
        var i = 0
        while (bb.hasRemaining()) {
          buffer.+=(bb.get)
          i += 1
        }
        i
      }
    }
    writeCSVToChannel(
      table,
      ch,
      batchSize,
      columnSeparator,
      quoteChar,
      recordSeparator,
      charset
    )
    new String(buffer.toArray, charset)
  }
  def writeCSVToChannel(
      table: Table,
      channel: WritableByteChannel,
      batchSize: Int = 10000,
      columnSeparator: Char = ',',
      quoteChar: Char = '"',
      recordSeparator: String = "\r\n",
      charset: Charset = StandardCharsets.UTF_8
  ): Unit = {

    def toStringColumns(rowIdx: Long, len: Int): Seq[Array[String]] = {
      0 until table.numCols map { colIdx =>
        val column = table.columns(colIdx)
        column.tpe match {
          case _: DateTimeColumnType =>
            Scope.unsafe { implicit scope =>
              column.values
                .slice(0, rowIdx, rowIdx + len, 1L)
                .toLongArray
                .map(l =>
                  java.time.Instant
                    .ofEpochMilli(l)
                    .toString
                )
            }

          case _: BooleanColumnType =>
            Scope.unsafe { implicit scope =>
              column.values
                .slice(0, rowIdx, rowIdx + len, 1)
                .toLongArray
                .map(l => (l == 1).toString)
            }

          case TextColumnType(_, pad, vocabulary) =>
            Scope.unsafe { implicit scope =>
              import lamp.saddle._
              val reverseVocabulary = vocabulary.map(_.map(_.swap))
              val rows =
                column.values.slice(0, rowIdx, rowIdx + len, 1).toLongMat
              rows.rows.map { row =>
                val str =
                  if (row.length == 0) (null: String)
                  else
                    row
                      .filter(_ != pad)
                      .map(l =>
                        reverseVocabulary.map(_.apply(l)).getOrElse(l.toChar)
                      )
                      .toArray
                      .mkString

                str
              }.toArray
            }

          case I64ColumnType =>
            val m = Scope.unsafe { implicit scope =>
              column.values
                .slice(0, rowIdx, rowIdx + len, 1)
                .toLongArray
                .map(_.toString)
            }
            m
          case F32ColumnType =>
            val m = Scope.unsafe { implicit scope =>
              column.values
                .slice(0, rowIdx, rowIdx + len, 1)
                .toFloatArray
                .map(_.toString)
            }
            m
          case F64ColumnType =>
            val m = Scope.unsafe { implicit scope =>
              column.values
                .slice(0, rowIdx, rowIdx + len, 1)
                .toDoubleArray
                .map(_.toString)
            }
            m
        }
      }
    }

    def quote(s: String) =
      if (s.contains(columnSeparator)) s"$quoteChar$s$quoteChar" else s

    val numRows = table.numRows
    val columnSeparatorStr = columnSeparator.toString
    channel.write(
      ByteBuffer.wrap(
        (table.colNames.toSeq
          .map(quote)
          .mkString(columnSeparatorStr)
          + recordSeparator)
          .getBytes(charset)
      )
    )
    0L.until(numRows, batchSize).foreach { case rowIdx =>
      val actualBatchSize = math.min(batchSize, numRows - rowIdx)
      val rows = toStringColumns(rowIdx, actualBatchSize.toInt)
      val renderedBatch = rows.transpose.map { row =>
        row.map(quote).mkString(columnSeparatorStr) + recordSeparator
      }.mkString
      val asByte = renderedBatch.getBytes(charset)
      channel.write(ByteBuffer.wrap(asByte))
    }

  }
  def readHeterogeneousFromCSVFile(
      columnTypes: Seq[(Int, ColumnDataType)],
      file: File,
      device: Device = lamp.CPU,
      charset: CharsetDecoder = makeAsciiSilentCharsetDecoder,
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
      charset: CharsetDecoder = makeAsciiSilentCharsetDecoder,
      fieldSeparator: Char = ',',
      quoteChar: Char = '"',
      recordSeparator: String = "\r\n",
      maxLines: Long = Long.MaxValue,
      header: Boolean = false,
      bufferSize: Int = 8192
  )(implicit
      scope: Scope
  ): Either[String, Table] = {

    val sortedColumnTypes = columnTypes
      .sortBy(_._1)
      .toArray

    val locs = sortedColumnTypes.map(_._1)

    val callback = new ColumnBufferCallback(maxLines, header, sortedColumnTypes)

    val error: Option[String] = org.saddle.io.csv.parse(
      channel = channel,
      callback = callback,
      bufferSize = bufferSize,
      charset = charset,
      fieldSeparator = fieldSeparator,
      quoteChar = quoteChar,
      recordSeparator = recordSeparator
    )

    error match {
      case Some(error) => Left(error)
      case None =>
        val colIndex = if (header) Some(callback.headerFields) else None

        if (locs.length > 0 && callback.headerLocFields != locs.length) {

          Left(
            s"Header line to short for given locs: ${locs.mkString("[", ", ", "]")}. Header line: ${callback.allHeaderFields
              .mkString("[", ", ", "]")}"
          )
        } else {
          val columns =
            callback.bufdata.zip(sortedColumnTypes).zipWithIndex map {
              case ((b, (_, tpe)), idx) =>
                val sten = tpe.copyBufferToSTen(b.asInstanceOf[tpe.Buf])
                val ondevice = if (device != CPU) {
                  device.to(sten)
                } else sten
                val name = colIndex.map(_.apply(idx))
                (name, Column(ondevice, tpe, None))
            }
          import org.saddle._
          Right(
            Table(
              columns.map(_._2).toVector,
              columns
                .map(_._1)
                .zipWithIndex
                .map { case (maybe, idx) => maybe.getOrElse(s"V$idx") }
                .toVector
                .toIndex
            )
          )

        }
    }
  }

  private[lamp] class ColumnBufferCallback(
      maxLines: Long,
      header: Boolean,
      columnTypes: Array[(Int, ColumnDataType)]
  ) extends Callback {

    val locs = columnTypes.map(_._1)
    private val locsIdx = org.saddle.Index(locs)

    val headerFields = scala.collection.mutable.ArrayBuffer[String]()
    val allHeaderFields = scala.collection.mutable.ArrayBuffer[String]()

    var headerAllFields = 0
    var headerLocFields = 0

    var bufdata: Array[_] = columnTypes.map { case (_, tpe) =>
      tpe.allocateBuffer()
    }

    val types: Array[ColumnDataType] = columnTypes.map(_._2)

    private val emptyLoc = locs.length == 0

    private final def add(s: Array[Char], from: Int, to: Int, buf: Int) = {
      val tpe0: ColumnDataType = types(buf)
      val buf0 = bufdata(buf)

      tpe0.tpeByte match {
        case 0 =>
          val tpe = tpe0.asInstanceOf[DateTimeColumnType]
          tpe.parseIntoBuffer(s, from, to, buf0.asInstanceOf[tpe.Buf])
        case 1 =>
          val tpe = tpe0.asInstanceOf[BooleanColumnType]
          tpe.parseIntoBuffer(s, from, to, buf0.asInstanceOf[tpe.Buf])
        case 2 =>
          val tpe = tpe0.asInstanceOf[TextColumnType]
          tpe.parseIntoBuffer(s, from, to, buf0.asInstanceOf[tpe.Buf])
        case 3 =>
          val tpe = I64ColumnType
          tpe.parseIntoBuffer(s, from, to, buf0.asInstanceOf[tpe.Buf])
        case 4 =>
          val tpe = F32ColumnType
          tpe.parseIntoBuffer(s, from, to, buf0.asInstanceOf[tpe.Buf])
        case 5 =>
          val tpe = F64ColumnType
          tpe.parseIntoBuffer(s, from, to, buf0.asInstanceOf[tpe.Buf])
      }

    }

    private var loc = 0
    private var line = 0L
    def apply(
        s: Array[Char],
        from: Array[Int],
        to: Array[Int],
        len: Int
    ): org.saddle.io.csv.Control = {
      var i = 0

      var error = false
      var errorString = ""

      if (len == -2) {
        error = true
        errorString =
          s"Unclosed quote after line $line (not necessarily in that line)"
      }

      while (i < len && line < maxLines && !error) {
        val fromi = from(i)
        val toi = to(i)
        val ptoi = math.abs(toi)
        if (line == 0) {
          allHeaderFields.append(new String(s, fromi, ptoi - fromi))
          headerAllFields += 1
          if (emptyLoc || locsIdx.contains(loc)) {
            headerLocFields += 1
          }
        }

        if (emptyLoc || locsIdx.contains(loc)) {
          if (header && line == 0) {
            headerFields.append(new String(s, fromi, ptoi - fromi))
          } else {
            if (loc >= headerAllFields) {
              error = true
              errorString =
                s"Too long line ${line + 1} (1-based). Expected $headerAllFields fields, got ${loc + 1}."
            } else {
              add(s, fromi, ptoi, loc)
            }
          }
        }

        if (toi < 0) {
          if (line == 0 && !emptyLoc && headerLocFields != locs.length) {
            error = true
            errorString =
              s"Header line to short for given locs: ${locs.mkString("[", ", ", "]")}. Header line: ${allHeaderFields
                .mkString("[", ", ", "]")}"
          }
          if (loc < headerAllFields - 1) {
            error = true
            errorString =
              s"Too short line ${line + 1} (1-based). Expected $headerAllFields fields, got ${loc + 1}."
          }

          loc = 0
          line += 1
        } else loc += 1
        i += 1
      }

      if (error) org.saddle.io.csv.Error(errorString)
      else if (line >= maxLines) org.saddle.io.csv.Done
      else org.saddle.io.csv.Next
    }

  }

}
