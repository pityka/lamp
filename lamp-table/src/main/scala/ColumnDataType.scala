package lamp.table

import lamp._
import lamp.io.csv.Buffer
import org.saddle.scalar.ScalarTagLong
import org.saddle.scalar.ScalarTagFloat
import org.saddle.scalar.ScalarTagDouble


sealed trait ColumnDataType {
  type Buf
  def allocateBuffer(): Buf
  def parseIntoBuffer(string: String, buffer: Buf): Unit
  def copyBufferToSTen(buf: Buf)(implicit scope: Scope): STen

}
final case class DateTimeColumnType(
    parse: String => Long = DateTimeColumnType.parse _
) extends ColumnDataType {
  override def toString = "DT"
  type Buf = Buffer[Long]
  def allocateBuffer() = Buffer.empty[Long](1024)
  def parseIntoBuffer(string: String, buffer: Buf): Unit =
    if (string == "na" || string == "NA") buffer.+=(ScalarTagLong.missing)
    else buffer.+=(parse(string))

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
object DateTimeColumnType {
  def parse(s: String): Long = java.time.Instant.parse(s).toEpochMilli()
}
final case class BooleanColumnType(
    isTrue: String => Boolean = BooleanColumnType.parse _
) extends ColumnDataType {

  override def toString = "B"
  type Buf = Buffer[Long]
  def allocateBuffer() = Buffer.empty[Long](1024)
  def parseIntoBuffer(string: String, buffer: Buf): Unit =
    if (string == "na" || string == "NA") buffer.+=(ScalarTagLong.missing)
    else buffer.+=(if (isTrue(string)) 1L else 0L)

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
object BooleanColumnType {
  def parse(s: String) =
    s == "true" || s == "T" || s == "True" || s == "TRUE" || s == "yes" || s == "Yes" || s == "Yes" || s.trim
      .toLowerCase() == "yes" || s.trim.toLowerCase() == "true"
}
final case class TextColumnType(
    maxLength: Int,
    pad: Long,
    vocabulary: Option[Map[Char, Long]]
) extends ColumnDataType {

  override def toString = "TXT"

  def tokenize(string: String): Array[Long] = string
    .map(char => vocabulary.map(_.apply(char)).getOrElse(char.toLong))
    .toArray
    .take(maxLength)
    .padTo(maxLength, pad)

  val missing = Array.ofDim[Long](maxLength).map(_ => ScalarTagLong.missing)

  /* (tokens, mask) */
  type Buf = Buffer[Array[Long]]
  def allocateBuffer() = Buffer.empty[Array[Long]](1024)
  def parseIntoBuffer(string: String, buffer: Buf): Unit =
    if (string == "na" || string == "NA") buffer.+=(missing)
    else
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
case object I64ColumnType extends ColumnDataType {

  override def toString = "I64"
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
case object F32ColumnType extends ColumnDataType {

  override def toString = "F32"
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
case object F64ColumnType extends ColumnDataType {

  override def toString = "F64"
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
