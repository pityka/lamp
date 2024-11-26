package lamp.io

import java.nio.channels.ReadableByteChannel
import lamp.Device
import java.nio.charset.CharsetDecoder
import org.saddle.scalar.ScalarTagDouble
import lamp.{STen, Sc}
import lamp.DoublePrecision
import lamp.CPU
import lamp.Scope
import org.saddle.scalar.ScalarTagLong
import org.saddle.scalar.ScalarTagFloat
import org.saddle.Buffer
import java.io.File

/** This package provides methods to read CSV formatted data into STen tensors
  *
  * The data is first read into to a regular JVM array, then transferred to
  * off-heap memory. The total tensor size may be larger than what a single JVM
  * array can hold.
  */
package object csv {
  def makeAsciiSilentCharsetDecoder =
    org.saddle.io.csv.makeAsciiSilentCharsetDecoder

  def readLongFromChannel(
      channel: ReadableByteChannel,
      device: Device,
      cols: Seq[Int] = Nil,
      fieldSeparator: Char = ',',
      quoteChar: Char = '"',
      recordSeparator: String = "\r\n",
      maxLines: Long = Long.MaxValue,
      header: Boolean = false,
      charset: CharsetDecoder = org.saddle.io.csv.makeAsciiSilentCharsetDecoder,
      bufferSize: Int = 8192
  )(implicit
      scope: Scope
  ): Either[String, (Option[List[String]], STen)] =
    readFromChannel(
      4,
      channel,
      device,
      cols,
      fieldSeparator,
      quoteChar,
      recordSeparator,
      maxLines,
      header,
      charset,
      bufferSize
    )
  def readFloatFromChannel(
      channel: ReadableByteChannel,
      device: Device,
      cols: Seq[Int] = Nil,
      fieldSeparator: Char = ',',
      quoteChar: Char = '"',
      recordSeparator: String = "\r\n",
      maxLines: Long = Long.MaxValue,
      header: Boolean = false,
      charset: CharsetDecoder = org.saddle.io.csv.makeAsciiSilentCharsetDecoder,
      bufferSize: Int = 8192
  )(implicit
      scope: Scope
  ): Either[String, (Option[List[String]], STen)] =
    readFromChannel(
      6,
      channel,
      device,
      cols,
      fieldSeparator,
      quoteChar,
      recordSeparator,
      maxLines,
      header,
      charset,
      bufferSize
    )
  def readDoubleFromChannel(
      channel: ReadableByteChannel,
      device: Device,
      cols: Seq[Int] = Nil,
      fieldSeparator: Char = ',',
      quoteChar: Char = '"',
      recordSeparator: String = "\r\n",
      maxLines: Long = Long.MaxValue,
      header: Boolean = false,
      charset: CharsetDecoder = org.saddle.io.csv.makeAsciiSilentCharsetDecoder,
      bufferSize: Int = 8192
  )(implicit
      scope: Scope
  ): Either[String, (Option[List[String]], STen)] =
    readFromChannel(
      7,
      channel,
      device,
      cols,
      fieldSeparator,
      quoteChar,
      recordSeparator,
      maxLines,
      header,
      charset,
      bufferSize
    )

  def readLongFromFile(
      file: File,
      device: Device,
      cols: Seq[Int] = Nil,
      fieldSeparator: Char = ',',
      quoteChar: Char = '"',
      recordSeparator: String = "\r\n",
      maxLines: Long = Long.MaxValue,
      header: Boolean = false,
      charset: CharsetDecoder = org.saddle.io.csv.makeAsciiSilentCharsetDecoder,
      bufferSize: Int = 8192
  )(implicit
      scope: Scope
  ): Either[String, (Option[List[String]], STen)] =
    readFromFile(
      4,
      file,
      device,
      cols,
      fieldSeparator,
      quoteChar,
      recordSeparator,
      maxLines,
      header,
      charset,
      bufferSize
    )
  def readFloatFromFile(
      file: File,
      device: Device,
      cols: Seq[Int] = Nil,
      fieldSeparator: Char = ',',
      quoteChar: Char = '"',
      recordSeparator: String = "\r\n",
      maxLines: Long = Long.MaxValue,
      header: Boolean = false,
      charset: CharsetDecoder = org.saddle.io.csv.makeAsciiSilentCharsetDecoder,
      bufferSize: Int = 8192
  )(implicit
      scope: Scope
  ): Either[String, (Option[List[String]], STen)] =
    readFromFile(
      6,
      file,
      device,
      cols,
      fieldSeparator,
      quoteChar,
      recordSeparator,
      maxLines,
      header,
      charset,
      bufferSize
    )
  def readDoubleFromFile(
      file: File,
      device: Device,
      cols: Seq[Int] = Nil,
      fieldSeparator: Char = ',',
      quoteChar: Char = '"',
      recordSeparator: String = "\r\n",
      maxLines: Long = Long.MaxValue,
      header: Boolean = false,
      charset: CharsetDecoder = org.saddle.io.csv.makeAsciiSilentCharsetDecoder,
      bufferSize: Int = 8192
  )(implicit
      scope: Scope
  ): Either[String, (Option[List[String]], STen)] =
    readFromFile(
      7,
      file,
      device,
      cols,
      fieldSeparator,
      quoteChar,
      recordSeparator,
      maxLines,
      header,
      charset,
      bufferSize
    )

  def readFromFile(
      scalarType: Byte,
      file: File,
      device: Device,
      cols: Seq[Int] = Nil,
      fieldSeparator: Char = ',',
      quoteChar: Char = '"',
      recordSeparator: String = "\r\n",
      maxLines: Long = Long.MaxValue,
      header: Boolean = false,
      charset: CharsetDecoder = org.saddle.io.csv.makeAsciiSilentCharsetDecoder,
      bufferSize: Int = 8192
  )(implicit
      scope: Scope
  ): Either[String, (Option[List[String]], STen)] = {
    val fis = new java.io.FileInputStream(file)
    val channel = fis.getChannel
    try {
      readFromChannel(
        scalarType,
        channel,
        device,
        cols,
        fieldSeparator,
        quoteChar,
        recordSeparator,
        maxLines,
        header,
        charset,
        bufferSize
      )
    } finally {
      fis.close
    }
  }

  /** Parse CSV files according to RFC 4180
    *
    * @param scalarType
    *   4=Long, 6=Float, 7=Double
    * @param channel
    * @param device
    * @param cols
    *   The column offsets to parse (if empty, parse everything)
    * @param fieldSeparator
    *   The separator; default is comma
    * @param quoteChar
    *   Within matching quotes, treat separChar as normal char; default is
    *   double-quote
    * @param recordSeparator
    *   Record separator (line ending)
    * @param maxLines
    *   The maximum number of records that will be read from the file. Includes
    *   header.
    * @param header
    *   indicates whether the first line should be set aside
    * @param charset
    * @param bufferSize
    */
  def readFromChannel(
      scalarType: Byte,
      channel: ReadableByteChannel,
      device: Device,
      cols: Seq[Int] = Nil,
      fieldSeparator: Char = ',',
      quoteChar: Char = '"',
      recordSeparator: String = "\r\n",
      maxLines: Long = Long.MaxValue,
      header: Boolean = false,
      charset: CharsetDecoder = org.saddle.io.csv.makeAsciiSilentCharsetDecoder,
      bufferSize: Int = 8192
  )(implicit
      scope: Scope
  ): Either[String, (Option[List[String]], STen)] = {
    val (copy) = typeSpecificLambdas(scalarType)
    val locs = Set(cols: _*).toArray[Int].sorted

    val callback = new BufferCallback(locs, maxLines, header, scalarType)

    val done: Option[String] = org.saddle.io.csv.parse(
      channel = channel,
      callback = callback,
      bufferSize = bufferSize,
      charset = charset,
      fieldSeparator = fieldSeparator,
      quoteChar = quoteChar,
      recordSeparator = recordSeparator
    )

    done match {
      case Some(error) => Left(error)
      case None =>
        val colIndex = if (header) Some(callback.headerFields) else None

        if (locs.length > 0 && callback.headerLocFields != locs.length) {

          Left(
            s"Header line to short for given locs: ${locs.mkString("[", ", ", "]")}. Header line: ${callback.allHeaderFields
                .mkString("[", ", ", "]")}"
          )
        } else {
          val sten = {
            val arrays = callback.bufdata.toArrays
            Scope { implicit scope =>
              STen
                .cat(arrays.map { array => copy(array) }, 0)
                .reshape(-1L, callback.headerLocFields.toLong)
            }
          }
          val t2 = if (device != CPU) {
            device.to(sten)
          } else sten
          Right((colIndex.map(_.toList), t2))
        }
    }

  }

  private[lamp] class BufferCallback(
      locs: Array[Int],
      maxLines: Long,
      header: Boolean,
      scalarType: Byte
  ) extends org.saddle.io.csv.Callback {

    private val locsIdx = org.saddle.Index(locs)

    val headerFields = scala.collection.mutable.ArrayBuffer[String]()
    val allHeaderFields = scala.collection.mutable.ArrayBuffer[String]()

    val bufdata = scalarType match {
      case 4 => Buffer.empty[Long](1024)
      case 6 => Buffer.empty[Float](1024)
      case 7 => Buffer.empty[Double](1024)
    }

    var headerAllFields = 0
    var headerLocFields = 0

    private val emptyLoc = locs.length == 0

    private var loc = 0
    private var line = 0L

    def apply(
        s: Array[Char],
        from: Array[Int],
        to: Array[Int],
        len: Int,
        eol: Array[Int]
    ): org.saddle.io.csv.Control = {
      scalarType match {
        case 4 => applyLong(s, from, to, len, eol)
        case 6 => applyFloat(s, from, to, len, eol)
        case 7 => applyDouble(s, from, to, len, eol)
      }
    }
    private def applyLong(
        s: Array[Char],
        from: Array[Int],
        to: Array[Int],
        len: Int,
        eol: Array[Int]
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
              import scala.Predef.{wrapRefArray => _}
              bufdata
                .asInstanceOf[Buffer[Long]]
                .+=(ScalarTagLong.parse(s, fromi, ptoi))
            }
          }
        }

        if (toi < 0 || eol(i) < 0) {
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
    private def applyFloat(
        s: Array[Char],
        from: Array[Int],
        to: Array[Int],
        len: Int,
        eol: Array[Int]
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
              import scala.Predef.{wrapRefArray => _}
              bufdata
                .asInstanceOf[Buffer[Float]]
                .+=(ScalarTagFloat.parse(s, fromi, ptoi))
            }
          }
        }

        if (toi < 0 || eol(i) < 0) {
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
    private def applyDouble(
        s: Array[Char],
        from: Array[Int],
        to: Array[Int],
        len: Int,
        eol: Array[Int]
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
              import scala.Predef.{wrapRefArray => _}
              bufdata
                .asInstanceOf[Buffer[Double]]
                .+=(ScalarTagDouble.parse(s, fromi, ptoi))
            }
          }
        }

        if (toi < 0 || eol(i) < 0) {
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

  private[csv] def typeSpecificLambdas[S: Sc](scalarType: Byte) =
    scalarType match {
      case 4 =>
        val copy = (ar: Array[_]) =>
          STen
            .fromLongArray(ar.asInstanceOf[Array[Long]], List(ar.length), CPU)

        (copy)
      case 6 =>
        val copy = (ar: Array[_]) =>
          STen
            .fromFloatArray(ar.asInstanceOf[Array[Float]], List(ar.length), CPU)

        (copy)
      case 7 =>
        val copy = (ar: Array[_]) =>
          STen
            .fromDoubleArray(
              ar.asInstanceOf[Array[Double]],
              List(ar.length),
              CPU,
              DoublePrecision
            )

        (copy)

    }

}
