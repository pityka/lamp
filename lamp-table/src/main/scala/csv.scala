package lamp.table

import lamp._
import java.nio.channels.ReadableByteChannel
import lamp.io.csv.asciiSilentCharsetDecoder
import java.nio.charset.CharsetDecoder
import java.io.File

object csv {
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
