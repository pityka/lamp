package lamp.data
import java.nio.ByteOrder
import java.nio.ByteBuffer
import java.nio.channels.WritableByteChannel
import java.io.File
import cats.effect.IO
import cats.effect.Resource
import java.io.FileOutputStream
import lamp.nn.GenericModule
import lamp.STen
import lamp.Scope
import java.nio.channels.Channels

/** Serializes tensors
  *
  * This format is similar to the ONNX external tensor serialization format, but
  * it uses JSON rather then protobuf.
  *
  * Format specification
  * ==================== Sequences of tensors are serialized into a JSON
  * descriptor and a data blob. The schema of the descriptor is the case class
  * lamp.data.schemas.TensorList. The location field in this schema holds a path
  * to the data blob. If this the location a relative POSIX then it is relative
  * to the file path where the descriptor itself is written.
  *
  * The descriptor may be embedded into larger JSON structures.
  *
  * The data blob itself is the raw data in little endian byte order. Floating
  * point is IEEE-754. The descriptor specifies the byte offset and byte length
  * of the tensors inside the data blob. As such, the data blob contains no
  * framing or other control bytes, but it may contain padding bytes between
  * tensors.
  */
object Writer {

  def writeTensorDataAndMakeDescriptor(
      tensors: Seq[STen],
      location: String,
      dataChannel: WritableByteChannel,
      initialByteOffset: Long,
      bufferSize: Int
  ) = {
    writeTensorDataIntoChannel(tensors, dataChannel, bufferSize).map {
      offsets =>
        val tensorDescriptors =
          offsets.zip(tensors).map { case ((offset, length, _), tensor) =>
            assert(length > 0, s"length is $length")
            assert(offset >= 0, s"offset is $offset")
            schemas.TensorDescriptor(
              dims = tensor.shape,
              dataType = tensor.scalarTypeByte,
              byteOffset = offset,
              byteLength = length
            )
          }

        schemas.TensorList(
          tensorDescriptors,
          location = location,
          byteOffset = initialByteOffset,
          byteLength = offsets.map(_._3).sum
        )

    }
  }

  private def tensorToArray(tensor0: STen, start: Long, end: Long) =
    Scope.leak { implicit scope =>
      val section = tensor0.view(-1).slice(0, start, end, 1)
      val t = if (section.isCPU) section else section.copyToDevice(lamp.CPU)

      val array = (t.scalarTypeByte match {
        case 7 =>
          val arr = Array.ofDim[Double]((end - start).toInt)
          assert(t.value.copyToDoubleArray(arr))
          arr
        case 6 =>
          val arr = Array.ofDim[Float]((end - start).toInt)
          assert(t.value.copyToFloatArray(arr))

          arr
        case 4 =>
          val arr = Array.ofDim[Long]((end - start).toInt)
          assert(t.value.copyToLongArray(arr))
          arr
      })

      array

    }

  private[data] def width(scalarTagByte: Byte) = (scalarTagByte: Byte) match {
    case 7     => Right(8)
    case 6     => Right(4)
    case 4     => Right(8)
    case other => Left(s"Type $other not supported.")
  }

  /** Returns pair of (data length, total bytes written). Total bytes is data +
    * pad. Pad pads to multiple of 8.
    */
  private def writeTensorIntoChannel(
      tensor: STen,
      channel: WritableByteChannel,
      bufferSize: Int
  ): Either[String, (Long, Long)] = {
    width(tensor.scalarTypeByte).map { width =>
      val elems = tensor.numel
      val bL = bufferSize.toLong
      val bb = ByteBuffer
        .allocate(bufferSize * width)
        .order(ByteOrder.LITTLE_ENDIAN)
      1L to (elems / bL + 1) foreach { i =>
        val start = (i - 1) * bufferSize
        val end = math.min(elems, i * bufferSize)

        val arr: Array[_] =
          tensorToArray(tensor, start, end)

        tensor.scalarTypeByte match {
          case 7 =>
            bb.asDoubleBuffer().put(arr.asInstanceOf[Array[Double]])
            bb.position(bb.position + arr.length * 8)

          case 6 =>
            bb.asFloatBuffer().put(arr.asInstanceOf[Array[Float]])
            bb.position(bb.position + arr.length * 4)

          case 4 =>
            bb.asLongBuffer().put(arr.asInstanceOf[Array[Long]])
            bb.position(bb.position + arr.length * 8)
        }

        writeFully(bb, channel)
      }
      val dataLength = elems * width.toLong
      val padLength = (8 - dataLength % 8) % 8
      if (padLength > 0) {
        val bb = ByteBuffer
          .allocate(padLength.toInt)
          .position(padLength.toInt)
          .asInstanceOf[ByteBuffer]
        writeFully(
          bb,
          channel
        )
      }
      (dataLength, dataLength + padLength)
    }

  }

  private def writeFully(bb: ByteBuffer, channel: WritableByteChannel) = {
    bb.flip
    while (bb.hasRemaining) {
      channel.write(bb)
    }
    bb.rewind
  }

  private def sequence[A, B](s: Seq[Either[A, B]]): Either[A, Seq[B]] =
    if (s.forall(_.isRight))
      Right(s.map(_.toOption.get))
    else s.find(_.isLeft).get.asInstanceOf[Left[A, Seq[B]]]

  /** Returns list of (offset, length) in bytes
    */
  def writeTensorDataIntoChannel(
      tensors: Seq[STen],
      channel: WritableByteChannel,
      bufferSize: Int
  ): Either[String, Seq[(Long, Long, Long)]] =
    sequence(tensors.map { case t =>
      writeTensorIntoChannel(t, channel, bufferSize)
    })
      .map { lengths =>
        (lengths
          .map(_._2)
          .scanLeft(0L)(_ + _)
          .dropRight(1) zip lengths.map(_._1) zip lengths.map(_._2)).map {
          case ((offset, dataLength), paddedLength) =>
            (offset, dataLength, paddedLength)
        }
      }

  def writeTensorsIntoFile(
      tensors: Seq[STen],
      file: File,
      bufferSize: Int = 16384
  ) = {
    val dataPath = new File(file.getAbsolutePath + ".data")
    val channel = Resource.make(IO {
      val descriptorChannel = new FileOutputStream(file, false).getChannel
      val dataChannel = new FileOutputStream(
        dataPath,
        false
      ).getChannel
      (descriptorChannel, dataChannel)
    })(v => IO { v._1.close; v._2.close })
    channel
      .use { case (descriptorChannel, dataChannel) =>
        IO {
          Writer
            .writeTensorDataAndMakeDescriptor(
              tensors = tensors,
              dataChannel = dataChannel,
              bufferSize = bufferSize,
              location = dataPath.getName,
              initialByteOffset = 0L
            )
            .map { descriptor =>
              com.github.plokhotnyuk.jsoniter_scala.core.writeToStream(
                descriptor,
                Channels.newOutputStream(descriptorChannel)
              )
            }
        }
      }
  }
  def writeCheckpoint[A, B](
      file: File,
      model: GenericModule[A, B],
      bufferSize: Int = 16384
  ) = {
    writeTensorsIntoFile(model.state.map(_._1.value), file, bufferSize)
  }

}
