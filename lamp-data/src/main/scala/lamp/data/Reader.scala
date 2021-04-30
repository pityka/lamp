package lamp.data
import org.saddle.{Buffer => _, _}
import org.saddle.scalar._
import java.nio._
import scala.util.{Left, Right, Either}
import java.nio.channels.ReadableByteChannel
import Writer._
import aten.Tensor
import aten.ATen
import cats.effect.IO
import java.io.FileInputStream
import cats.effect.Resource
import java.io.File
import lamp.Device
import lamp.SinglePrecision
import lamp.nn.GenericModule
import lamp.nn.Load
import lamp.Scope
import lamp.STen

object Reader {

  def parse[T: ST](size: Int, from: ByteBuffer): Either[String, Array[T]] =
    implicitly[ST[T]] match {
      case ScalarTagDouble =>
        Right {
          val to = DoubleBuffer.allocate(size)
          while (to.hasRemaining() && from.hasRemaining()) {
            to.put(from.getDouble)
          }
          to.array
        }
      case ScalarTagInt =>
        Right {
          val to = IntBuffer.allocate(size)
          while (to.hasRemaining() && from.hasRemaining()) {
            to.put(from.getInt)
          }
          to.array
        }
      case ScalarTagFloat =>
        Right {
          val to = FloatBuffer.allocate(size)
          while (to.hasRemaining() && from.hasRemaining()) {
            to.put(from.getFloat())
          }
          to.array
        }
      case ScalarTagLong =>
        Right {
          val to = LongBuffer.allocate(size)
          while (to.hasRemaining() && from.hasRemaining()) {
            to.put(from.getLong())
          }
          to.array
        }
      case ScalarTagByte =>
        Right {
          val to = ByteBuffer.allocate(size)
          while (to.hasRemaining() && from.hasRemaining()) {
            to.asInstanceOf[ByteBuffer].put(from.get)
          }
          to.array
        }
      case other => Left(s"Type $other not supported.")
    }

  def sequence[A, B](s: Seq[Either[A, B]]): Either[A, Seq[B]] =
    if (s.forall(_.isRight))
      Right(s.map(_.toOption.get))
    else s.find(_.isLeft).get.asInstanceOf[Left[A, Seq[B]]]

  def readFully(bb: ByteBuffer, channel: ReadableByteChannel) = {
    bb.clear
    var i = 0
    while (bb.hasRemaining && i >= 0) {
      i = channel.read(bb)
    }
    bb.flip
    i
  }

  def readHeaderFromChannel(channel: ReadableByteChannel) = {
    val magicAndVersion = Array.ofDim[Byte](8)
    val count = readFully(ByteBuffer.wrap(magicAndVersion), channel)
    val magic = String.valueOf(magicAndVersion.map(_.toChar), 0, 6)
    if (count < 8) Left("EOF")
    else {
      val major = magicAndVersion(6)
      val minor = magicAndVersion(7)
      if (major != 1 || minor != 0 || magic != "LAMP__")
        Left(
          s"Magic string is incorrect or version of file format not supported. Found in file: $magic / $major / $minor"
        )
      else {
        val headerLengthBB =
          ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN)
        readFully(headerLengthBB, channel)
        val headerLength = headerLengthBB.getInt
        val headerArray = Array.ofDim[Byte](headerLength)
        readFully(ByteBuffer.wrap(headerArray), channel)
        val header = new String(headerArray, "UTF-8")
        val descriptor = ujson.read(header).obj
        val version = descriptor("v").num.toInt
        if (version != 1)
          Left("Data layout not supported")
        else Right(descriptor)
      }
    }
  }

  def readTensorDataFromChannel[T: ST](
      channel: ReadableByteChannel,
      shape: List[Int],
      width: Int,
      device: Device
  ): Either[String, Tensor] = {
    val numel = if (shape.size == 0) 1 else shape.reduce(_ * _)
    val bb = ByteBuffer
      .allocate(width * numel)
      .order(ByteOrder.LITTLE_ENDIAN)
    readFully(bb, channel)
    parse[T](numel, bb).flatMap { data =>
      if (data.size != numel) {
        Left("Premature end of input")
      } else {
        Scope.leak { implicit scope =>
          val dopt = device.options(SinglePrecision)
          val topt = implicitly[ST[T]] match {
            case ScalarTagDouble => dopt.toDouble
            case ScalarTagFloat  => dopt.toFloat
            case ScalarTagLong   => dopt.toLong
          }
          val t = ATen.zeros(shape.map(_.toLong).toArray, topt.cpu.value)
          implicitly[ST[T]] match {
            case ScalarTagDouble => assert(t.copyFromDoubleArray(data))
            case ScalarTagFloat  => assert(t.copyFromFloatArray(data))
            case ScalarTagLong   => assert(t.copyFromLongArray(data))
          }
          val tdevice = t.to(topt.value, true, true)
          t.release
          Right(tdevice)
        }
      }
    }
  }

  def readTensorFromChannel(
      channel: ReadableByteChannel,
      device: Device
  ): Either[String, Tensor] = {
    readHeaderFromChannel(channel).flatMap { descriptor =>
      val datatype = descriptor(KEY_datatype).str
      val scalarTagByte = datatype match {
        case "double" => 7
        case "float"  => 6
        case "long"   => 4
      }

      width(scalarTagByte.toByte).flatMap { width =>
        val shape =
          descriptor
            .get(KEY_shape)
            .map(_.arr.map(_.num.toInt).toList)
        shape match {
          case None => Left("No shape")
          case Some(shape) =>
            datatype match {
              case "double" =>
                readTensorDataFromChannel[Double](channel, shape, width, device)
              case "float" =>
                readTensorDataFromChannel[Float](channel, shape, width, device)
              case "long" =>
                readTensorDataFromChannel[Long](channel, shape, width, device)
            }

        }
      }
    }
  }

  def readTensorsFromChannel(
      channel: ReadableByteChannel,
      device: Device
  ): Either[String, Seq[Tensor]] =
    Reader.sequence(
      Iterator
        .continually {
          readTensorFromChannel(channel, device)
        }
        .takeWhile(_.isRight)
        .toList
    )

  def readTensorsFromFile(
      file: File,
      device: Device
  ): Either[String, Seq[Tensor]] = {
    val fis = new FileInputStream(file)
    try {
      readTensorsFromChannel(fis.getChannel, device)
    } finally {
      fis.close
    }
  }

  class ByteChannel(src: ByteBuffer) extends ReadableByteChannel {
    def read(dst: ByteBuffer) = {
      var i = 0
      while (dst.hasRemaining() && src.hasRemaining()) {
        dst.put(src.get)
        i += 1
      }
      i
    }
    def isOpen(): Boolean = true
    def close(): Unit = ()
  }

  def readTensorFromArray(
      array: Array[Byte],
      device: Device
  ): Either[String, Tensor] =
    readTensorFromChannel(new ByteChannel(ByteBuffer.wrap(array)), device)

  def scalarTypeToScalarTag(t: Byte) = t match {
    case 4 => ScalarTagLong
    case 6 => ScalarTagFloat
    case 7 => ScalarTagDouble
  }

  def loadFromFile[A, B, M <: GenericModule[A, B]: Load](
      module: M with GenericModule[A, B],
      file: File,
      device: Device
  ) = {
    val channel = Resource.make(IO {
      val fis = new FileInputStream(file)
      fis.getChannel
    })(v => IO { v.close })
    loadFromChannel(module, channel, device)
  }
  def loadFromChannel[A, B, M <: GenericModule[A, B]: Load](
      module: M with GenericModule[A, B],
      channel: Resource[IO, ReadableByteChannel],
      device: Device
  ) = {

    channel
      .use { channel =>
        IO {
          Reader
            .readTensorsFromChannel(
              channel,
              device
            )
            .map { tensors =>
              Scope.root { implicit scope =>
                module.load(tensors.map(t => STen.owned(t)))
              }
            }
        }
      }

  }
}
