package lamp.data
import org.saddle.scalar._
import org.saddle._
import java.nio.ByteOrder
import java.nio.ByteBuffer
import java.nio.channels.WritableByteChannel
import aten.Tensor
import java.io.File
import cats.effect.IO
import cats.effect.Resource
import java.io.FileOutputStream
import lamp.nn.GenericModule

/** Binary serialization for Tensor with primitive Double, Float, Long
  *
  * The layout of binary format is as follows:
  * - The first 6 bytes are "LAMP__"
  * - The next unsigned byte is the major version
  * - The next unsigned byte is the minor version
  * - The next 4 bytes form a little endian integer as HEADER_LENGTH
  * - The next HEADER_LENGTH bytes form an UTF-8 string as the header.
  * - The header is a valid JSON object with the following fields:
  *   - v: numeric positive integer is the version of the header structure
  *   - shape: numeric array
  *   - datatype : string, either "double", "long", "int", "float", "byte"
  * - The header is padded with spaces (0x20) such that HEADER_LENGTH+12 is divisible by 16.
  *   The count of spaces are included in HEADER_LENGTH.
  * - The next width * shape.reduce(_ * _)  bytes form a little endian primitive array.
  */
object Writer {
  private def int(i: Int) = {
    ByteBuffer
      .allocate(4)
      .order(ByteOrder.LITTLE_ENDIAN)
      .putInt(i)
      .array
  }

  val KEY_datatype = "datatype"
  val KEY_shape = "shape"
  val KEY_v = "v"

  def createTensorDescriptor[T: ST](
      tensor: Tensor
  ) = dtype[T].right.map { dtype =>
    ujson
      .write(
        ujson.Obj(
          KEY_datatype -> dtype,
          KEY_shape -> ujson.Arr(tensor.sizes.map(l => ujson.Num(l)): _*),
          KEY_v -> 1
        )
      )
  }

  def createHeader(
      descriptor: Either[String, String]
  ): Either[String, Array[Byte]] = {
    descriptor.right.map { descriptorJson =>
      val descriptor = {
        val json = descriptorJson
          .getBytes("UTF-8")
        val jsonLength = json.length
        val usefulLength = jsonLength + 12
        val padding = usefulLength / 16 + 16 - usefulLength
        json ++ Array.fill(padding)(' '.toByte)
      }
      val magic = "LAMP__".getBytes("US-ASCII")
      val version = Array(1.toByte, 0.toByte)
      val headerLength = int(descriptor.length)
      val header = magic ++ version ++ headerLength ++ descriptor
      header
    }
  }

  private[data] def dtype[T: ST] = implicitly[ST[T]] match {
    case ScalarTagDouble => Right("double")
    case ScalarTagInt    => Right("int")
    case ScalarTagFloat  => Right("float")
    case ScalarTagLong   => Right("long")
    case ScalarTagByte   => Right("byte")
    case other           => Left(s"Type $other not supported.")
  }

  private[data] def width[T: ST] = implicitly[ST[T]] match {
    case ScalarTagDouble => Right(8)
    case ScalarTagInt    => Right(4)
    case ScalarTagFloat  => Right(4)
    case ScalarTagLong   => Right(8)
    case ScalarTagByte   => Right(1)
    case other           => Left(s"Type $other not supported.")
  }

  def put[@specialized(Double, Long, Int, Float, Byte) T: ST](
      t: Array[T],
      bb: ByteBuffer
  ) = implicitly[ST[T]] match {
    case ScalarTagDouble =>
      Right {
        var i = 0
        val n = t.length
        while (i < n) {
          bb.putDouble(t(i))
          i += 1
        }
      }
    case ScalarTagInt =>
      Right {
        var i = 0
        val n = t.length
        while (i < n) {
          bb.putInt(t(i).asInstanceOf[Int])
          i += 1
        }
      }
    case ScalarTagFloat =>
      Right {
        var i = 0
        val n = t.length
        while (i < n) {
          bb.putFloat(t(i).asInstanceOf[Float])
          i += 1
        }
      }
    case ScalarTagLong =>
      Right {
        var i = 0
        val n = t.length
        while (i < n) {
          bb.putLong(t(i).asInstanceOf[Long])
          i += 1
        }
      }
    case ScalarTagByte =>
      Right {
        var i = 0
        val n = t.length
        while (i < n) {
          bb.put(t(i).asInstanceOf[Byte])
          i += 1
        }
      }
    case other => Left(s"Type $other not supported.")
  }

  def tensorToArray[T: ST](tensor: Tensor, elem: Int) =
    (implicitly[ST[T]] match {
      case ScalarTagDouble =>
        val arr = Array.ofDim[Double](elem)
        tensor.copyToDoubleArray(arr)
        arr
      case ScalarTagFloat =>
        val arr = Array.ofDim[Float](elem)
        tensor.copyToFloatArray(arr)
        arr
      case ScalarTagLong =>
        val arr = Array.ofDim[Long](elem)
        tensor.copyToLongArray(arr)
        arr
    }).asInstanceOf[Array[T]]

  def writeTensorIntoChannel[T: ST](
      tensor: Tensor,
      channel: WritableByteChannel
  ): Either[String, Unit] = {
    val header = createHeader(createTensorDescriptor[T](tensor))
    header.right.flatMap { header =>
      width[T].right.map { width =>
        writeFully(ByteBuffer.wrap(header), channel)

        val elem = tensor.numel().toInt
        val arr: Array[T] =
          tensorToArray[T](tensor, elem)
        val bb = ByteBuffer
          .allocate(arr.length * width)
          .order(ByteOrder.LITTLE_ENDIAN)
        put(arr, bb)
        writeFully(bb, channel)
      }
    }
  }

  def writeFully(bb: ByteBuffer, channel: WritableByteChannel) = {
    bb.rewind
    while (bb.hasRemaining) {
      channel.write(bb)
    }
  }

  def writeTensorIntoArray[T: ST](
      tensor: Tensor
  ): Either[String, Array[Byte]] = {
    val header = createHeader(createTensorDescriptor(tensor))
    header.right.flatMap { header =>
      width[T].right.map { width =>
        val elem = tensor.numel.toInt
        val result =
          Array.ofDim[Byte](header.length + width * elem)
        System.arraycopy(header, 0, result, 0, header.length)
        val bb = ByteBuffer.wrap(result).order(ByteOrder.LITTLE_ENDIAN)
        bb.position(header.length)

        val ar: Array[T] =
          tensorToArray[T](tensor, elem)
        put(ar, bb)
        result
      }
    }
  }

  def writeTensorsIntoChannel(
      tensors: Seq[Tensor],
      channel: WritableByteChannel
  ): Either[String, Seq[Unit]] =
    Reader.sequence(tensors.map {
      case t =>
        val st = t.scalarType() match {
          case 4 => ScalarTagLong
          case 6 => ScalarTagFloat
          case 7 => ScalarTagDouble
        }
        writeTensorIntoChannel(t, channel)(st)
    })

  def writeCheckpoint[A, B](file: File, model: GenericModule[A, B]) = {
    val channel = Resource.make(IO {
      val fis = new FileOutputStream(file, false)
      fis.getChannel
    })(v => IO { v.close })
    channel
      .use { channel =>
        IO {
          Writer.writeTensorsIntoChannel(
            model.state
              .map(v => v._1.value.value),
            channel
          )
        }
      }
  }

}
