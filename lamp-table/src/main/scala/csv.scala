package lamp.table

import lamp._
import java.nio.channels.ReadableByteChannel
import lamp.io.csv.asciiSilentCharsetDecoder
import java.nio.charset.CharsetDecoder
import java.io.File
import java.nio.channels.WritableByteChannel
import java.nio.charset.Charset
import java.nio.charset.StandardCharsets
import java.nio.ByteBuffer
import org.saddle.Buffer

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
          +recordSeparator)
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
          (name, Column(ondevice, tpe, None))
      }
      if (columns.map(_._2.values.shape(0)).distinct.size != 1)
        Left(
          s"Uneven length ${columns.map(_._2.values.shape(0)).toVector} columns"
        )
      else {
        import org.saddle._
        Right(
          Table(
            columns.map(_._2).toVector,
            columns
              .map(_._1)
              .zipWithIndex
              .map { case (maybe, idx) => maybe.getOrElse(s"V$idx") }
              .toIndex
          )
        )
      }
    }

  }
}
