package lamp.data
import java.nio.channels.ReadableByteChannel
import cats.effect.IO
import java.io.FileInputStream
import java.io.File
import lamp.Device
import lamp.nn.GenericModule
import lamp.nn.Load
import lamp.Scope
import lamp.STen
import java.nio.channels.Channels
import java.nio.ByteBuffer

object Reader {

  def readTensorListDescriptorFromChannel(
      channel: ReadableByteChannel
  ): schemas.TensorList = {
    val is = Channels.newInputStream(channel)
    com.github.plokhotnyuk.jsoniter_scala.core
      .readFromStream[schemas.TensorList](is)
  }

  def readTensorListDescriptorFromFile(
      file: File
  ): schemas.TensorList = {
    val fis = new FileInputStream(file)
    try {
      readTensorListDescriptorFromChannel(fis.getChannel)
    } finally {
      fis.close
    }
  }

  def readSingleTensor(
      path: String,
      offset: Long,
      length: Long,
      scalarTypeByte: Byte,
      dims: Seq[Long],
      device: Device
  )(implicit scope: Scope) = Scope { implicit scope =>
    device.to(
      STen.fromFile(path, offset, length, scalarTypeByte).view(dims: _*)
    )
  }

  def readTensorData(
      descriptor: schemas.TensorList,
      pathOfDescriptor: File,
      device: Device
  )(implicit scope: Scope): Seq[STen] = {
    descriptor.tensors.map { td =>
      readSingleTensor(
        new File(pathOfDescriptor.getParent, td.location).getAbsolutePath(),
        td.byteOffset,
        td.byteLength,
        td.dataType,
        td.dims,
        device
      )

    }
  }

  def readTensorsFromFile(
      file: File,
      device: Device
  )(implicit scope: Scope): Seq[STen] = {
    val d = readTensorListDescriptorFromFile(file)
    readTensorData(d, file, device)
  }

  def loadFromFile[A, B, M <: GenericModule[A, B]: Load](
      module: M with GenericModule[A, B],
      file: File,
      device: Device
  ): IO[Unit] = {

    IO {
      Scope.root { implicit scope =>
        val tensors = Reader
          .readTensorsFromFile(
            file,
            device
          )
        module.load(tensors)
      }
    }

  }

  def readFully(bb: ByteBuffer, channel: ReadableByteChannel) = {
    bb.clear
    var i = 0
    while (bb.hasRemaining && i >= 0) {
      i = channel.read(bb)
    }
    bb.flip
    i
  }
}
