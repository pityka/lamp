package lamp.io

import java.nio.channels.ReadableByteChannel
import lamp.Device
import java.nio.charset.CharsetDecoder
import org.saddle.scalar.ScalarTagDouble
import lamp.STen
import lamp.DoublePrecision
import lamp.CPU
import lamp.Scope
import org.saddle.scalar.ScalarTagLong
import org.saddle.scalar.ScalarTagFloat
import java.io.File

/** This package provides methods to read CSV formatted data into STen tensors
  *
  * The data is first read into to a regular JVM array, then transferred to off-heap memory.
  * The total tensor size may be larger than what a single JVM array can hold.
  */
package object csv {
  val asciiSilentCharsetDecoder = org.saddle.io.csv.asciiSilentCharsetDecoder

  def readLongFromChannel(
      channel: ReadableByteChannel,
      device: Device,
      charset: CharsetDecoder = asciiSilentCharsetDecoder,
      cols: Seq[Int] = Nil,
      fieldSeparator: Char = ',',
      quoteChar: Char = '"',
      recordSeparator: String = "\r\n",
      maxLines: Long = Long.MaxValue,
      header: Boolean = false
  )(
      implicit scope: Scope
  ): Either[String, (Option[List[String]], STen)] =
    readFromChannel(
      4,
      channel,
      device,
      charset,
      cols,
      fieldSeparator,
      quoteChar,
      recordSeparator,
      maxLines,
      header
    )
  def readFloatFromChannel(
      channel: ReadableByteChannel,
      device: Device,
      charset: CharsetDecoder = asciiSilentCharsetDecoder,
      cols: Seq[Int] = Nil,
      fieldSeparator: Char = ',',
      quoteChar: Char = '"',
      recordSeparator: String = "\r\n",
      maxLines: Long = Long.MaxValue,
      header: Boolean = false
  )(
      implicit scope: Scope
  ): Either[String, (Option[List[String]], STen)] =
    readFromChannel(
      6,
      channel,
      device,
      charset,
      cols,
      fieldSeparator,
      quoteChar,
      recordSeparator,
      maxLines,
      header
    )
  def readDoubleFromChannel(
      channel: ReadableByteChannel,
      device: Device,
      charset: CharsetDecoder = asciiSilentCharsetDecoder,
      cols: Seq[Int] = Nil,
      fieldSeparator: Char = ',',
      quoteChar: Char = '"',
      recordSeparator: String = "\r\n",
      maxLines: Long = Long.MaxValue,
      header: Boolean = false
  )(
      implicit scope: Scope
  ): Either[String, (Option[List[String]], STen)] =
    readFromChannel(
      7,
      channel,
      device,
      charset,
      cols,
      fieldSeparator,
      quoteChar,
      recordSeparator,
      maxLines,
      header
    )

  def readLongFromFile(
      file: File,
      device: Device,
      charset: CharsetDecoder = asciiSilentCharsetDecoder,
      cols: Seq[Int] = Nil,
      fieldSeparator: Char = ',',
      quoteChar: Char = '"',
      recordSeparator: String = "\r\n",
      maxLines: Long = Long.MaxValue,
      header: Boolean = false
  )(
      implicit scope: Scope
  ): Either[String, (Option[List[String]], STen)] =
    readFromFile(
      4,
      file,
      device,
      charset,
      cols,
      fieldSeparator,
      quoteChar,
      recordSeparator,
      maxLines,
      header
    )
  def readFloatFromFile(
      file: File,
      device: Device,
      charset: CharsetDecoder = asciiSilentCharsetDecoder,
      cols: Seq[Int] = Nil,
      fieldSeparator: Char = ',',
      quoteChar: Char = '"',
      recordSeparator: String = "\r\n",
      maxLines: Long = Long.MaxValue,
      header: Boolean = false
  )(
      implicit scope: Scope
  ): Either[String, (Option[List[String]], STen)] =
    readFromFile(
      6,
      file,
      device,
      charset,
      cols,
      fieldSeparator,
      quoteChar,
      recordSeparator,
      maxLines,
      header
    )
  def readDoubleFromFile(
      file: File,
      device: Device,
      charset: CharsetDecoder = asciiSilentCharsetDecoder,
      cols: Seq[Int] = Nil,
      fieldSeparator: Char = ',',
      quoteChar: Char = '"',
      recordSeparator: String = "\r\n",
      maxLines: Long = Long.MaxValue,
      header: Boolean = false
  )(
      implicit scope: Scope
  ): Either[String, (Option[List[String]], STen)] =
    readFromFile(
      7,
      file,
      device,
      charset,
      cols,
      fieldSeparator,
      quoteChar,
      recordSeparator,
      maxLines,
      header
    )

  def readFromFile(
      scalarType: Byte,
      file: File,
      device: Device,
      charset: CharsetDecoder = asciiSilentCharsetDecoder,
      cols: Seq[Int] = Nil,
      fieldSeparator: Char = ',',
      quoteChar: Char = '"',
      recordSeparator: String = "\r\n",
      maxLines: Long = Long.MaxValue,
      header: Boolean = false
  )(
      implicit scope: Scope
  ): Either[String, (Option[List[String]], STen)] = {
    val fis = new java.io.FileInputStream(file)
    val channel = fis.getChannel
    try {
      readFromChannel(
        scalarType,
        channel,
        device,
        charset,
        cols,
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
  def readFromChannel(
      scalarType: Byte,
      channel: ReadableByteChannel,
      device: Device,
      charset: CharsetDecoder = asciiSilentCharsetDecoder,
      cols: Seq[Int] = Nil,
      fieldSeparator: Char = ',',
      quoteChar: Char = '"',
      recordSeparator: String = "\r\n",
      maxLines: Long = Long.MaxValue,
      header: Boolean = false
  )(
      implicit scope: Scope
  ): Either[String, (Option[List[String]], STen)] = {
    val (append, allocBuffer, copy) = scalarType match {
      case 4 =>
        val copy = (ar: Array[_]) =>
          STen
            .fromLongArray(ar.asInstanceOf[Array[Long]], List(ar.length), CPU)
        val allocBuffer = () => Buffer.empty[Long](1024)
        val parse = (s: String, buffer: Buffer[_]) => {
          buffer.asInstanceOf[Buffer[Long]].+=(ScalarTagLong.parse(s))
        }
        (parse, allocBuffer, copy)
      case 6 =>
        val copy = (ar: Array[_]) =>
          STen
            .fromFloatArray(ar.asInstanceOf[Array[Float]], List(ar.length), CPU)
        val allocBuffer = () => Buffer.empty[Float](1024)
        val parse = (s: String, buffer: Buffer[_]) => {
          buffer.asInstanceOf[Buffer[Float]].+=(ScalarTagFloat.parse(s))
        }
        (parse, allocBuffer, copy)
      case 7 =>
        val copy = (ar: Array[_]) =>
          STen
            .fromDoubleArray(
              ar.asInstanceOf[Array[Double]],
              List(ar.length),
              CPU,
              DoublePrecision
            )
        val allocBuffer = () => Buffer.empty[Double](1024)
        val parse = (s: String, buffer: Buffer[_]) => {
          buffer.asInstanceOf[Buffer[Double]].+=(ScalarTagDouble.parse(s))
        }
        (parse, allocBuffer, copy)

    }
    val source = org.saddle.io.csv
      .readChannel(channel, bufferSize = 65536, charset = charset)

    var locs = Set(cols: _*).toArray[Int].sorted

    var bufdata: Seq[Buffer[_]] = null

    def prepare(headerLength: Int) = {
      if (locs.length == 0) locs = (0 until headerLength).toArray
      bufdata = for { _ <- locs.toSeq } yield allocBuffer()
    }

    def addToBuffer(s: String, buf: Int) = {
      append(s, bufdata(buf))
    }

    val done = org.saddle.io.csv.parseFromIteratorCallback(
      source,
      cols,
      fieldSeparator,
      quoteChar,
      recordSeparator,
      maxLines,
      header
    )(prepare, addToBuffer)

    done.flatMap { colIndex =>
      val columns = bufdata map { b =>
        val arrays = b.toArrays
        Scope { implicit scope =>
          STen.cat(arrays.map { array => copy(array) }, 0)
        }
      }
      if (columns.map(_.shape(0)).distinct.size != 1)
        Left(s"Uneven length ${columns.map(_.shape(0)).toVector} columns")
      else {
        val stacked = STen.stack(columns, 1)
        val t2 = if (device != CPU) {
          device.to(stacked)
        } else stacked
        Right((colIndex.map(_.toList), t2))
      }
    }

  }
}
