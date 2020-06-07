package lamp.data
import org.saddle.{Buffer => _, _}
import org.saddle.scalar._
import org.saddle.order._
import java.nio._
import scala.util.{Left, Right, Either}
import java.nio.channels.ReadableByteChannel
import Writer._
import org.saddle.index.IndexIntRange
import aten.Tensor
import aten.TensorOptions
import aten.ATen

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
      Right(s.map(_.right.get))
    else s.find(_.isLeft).get.asInstanceOf[Left[A, Seq[B]]]

  def readFully(bb: ByteBuffer, channel: ReadableByteChannel) = {
    bb.clear
    var i = 0
    while (bb.hasRemaining && i >= 0) {
      i = channel.read(bb)
    }
    bb.flip
  }

  def readHeaderFromChannel[T: ST](channel: ReadableByteChannel) =
    dtype[T].right.flatMap { expectedDataType =>
      val magicAndVersion = Array.ofDim[Byte](8)
      readFully(ByteBuffer.wrap(magicAndVersion), channel)
      val magic = String.valueOf(magicAndVersion.map(_.toChar), 0, 6)
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
        val datatype = descriptor(KEY_datatype).str
        val version = descriptor("v").num.toInt
        if (version != 1 || datatype != expectedDataType)
          Left("Data layout not supported")
        else Right(descriptor)
      }
    }

  def readTensorDataFromChannel[T: ST](
      channel: ReadableByteChannel,
      shape: List[Int],
      width: Int
  ): Either[String, Tensor] = {
    val numel = if (shape.size == 0) 1 else shape.reduce(_ * _)
    val bb = ByteBuffer
      .allocate(width * numel)
      .order(ByteOrder.LITTLE_ENDIAN)
    readFully(bb, channel)
    parse[T](numel, bb).right.flatMap { data =>
      if (data.size != numel) {
        Left("Premature end of input")
      } else {
        val topt = implicitly[ST[T]] match {
          case ScalarTagDouble => TensorOptions.dtypeDouble()
          case ScalarTagFloat  => TensorOptions.dtypeFloat()
          case ScalarTagLong   => TensorOptions.dtypeLong()
        }
        val t = ATen.zeros(shape.map(_.toLong).toArray, topt)
        implicitly[ST[T]] match {
          case ScalarTagDouble => t.copyFromDoubleArray(data)
          case ScalarTagFloat  => t.copyFromFloatArray(data)
          case ScalarTagLong   => t.copyFromLongArray(data)
        }
        Right(t)
      }
    }
  }

  def readTensorFromChannel[T: ST](
      channel: ReadableByteChannel
  ): Either[String, Tensor] = {
    readHeaderFromChannel[T](channel).right.flatMap { descriptor =>
      width[T].right.flatMap { width =>
        val shape =
          descriptor
            .get(KEY_shape)
            .map(_.arr.map(_.num.toInt).toList)
        shape match {
          case None        => Left("No shape")
          case Some(shape) => readTensorDataFromChannel(channel, shape, width)
        }
      }
    }
  }

  def readTensorsFromChannel(
      types: Seq[(ScalarTag[_])],
      channel: ReadableByteChannel
  ): Either[String, Seq[Tensor]] =
    Reader.sequence(types.map {
      case (st) =>
        readTensorFromChannel(channel)(st)
    })

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

  def readTensorFromArray[T: ST](
      array: Array[Byte]
  ): Either[String, Tensor] =
    readTensorFromChannel(new ByteChannel(ByteBuffer.wrap(array)))
}
