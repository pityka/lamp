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

 private[lamp]  def readTensorListDescriptorFromChannel(
      channel: ReadableByteChannel
  ): schemas.TensorList = {
    val is = Channels.newInputStream(channel)
    com.github.plokhotnyuk.jsoniter_scala.core
      .readFromStream[schemas.TensorList](is)
  }

 private[lamp]  def readTensorListDescriptorFromFile(
      file: File
  ): schemas.TensorList = {
    val fis = new FileInputStream(file)
    try {
      readTensorListDescriptorFromChannel(fis.getChannel)
    } finally {
      fis.close
    }
  }

  private[lamp] def readTensorData(
      descriptor: schemas.TensorList,
      pathOfDescriptor: File,
      device: Device,
      pin: Boolean
  )(implicit scope: Scope): Seq[STen] = {
    Scope { implicit scope =>
      STen
        .tensorsFromFile(
          path = new File(pathOfDescriptor.getParent, descriptor.location)
            .getAbsolutePath(),
          offset = descriptor.byteOffset,
          length = descriptor.byteLength,
          pin = pin,
          tensors = descriptor.tensors.map { td =>
            (td.dataType, td.byteOffset, td.byteLength)
          }.toList
        )
        .zip(descriptor.tensors)
        .map { case (t1, descr) =>
          device.to(t1.view(descr.dims: _*))
        }
    }

  }

 def readTensorsFromFile(
      file: File,
      device: Device,
      pin: Boolean
  )(implicit scope: Scope): Seq[STen] = {
    val d = readTensorListDescriptorFromFile(file)
    readTensorData(d, file, device, pin)
  }

  def loadFromFile[A, B, M <: GenericModule[A, B]: Load](
      module: M with GenericModule[A, B],
      file: File,
      device: Device,
      pin: Boolean
  ): IO[Unit] = {

    IO {
      Scope.root { implicit scope =>
        val tensors = Reader
          .readTensorsFromFile(
            file,
            device,
            pin
          )
        module.load(tensors)
      }
    }

  }

 private[lamp]  def readFully(bb: ByteBuffer, channel: ReadableByteChannel) = {
    bb.clear
    var i = 0
    while (bb.hasRemaining && i >= 0) {
      i = channel.read(bb)
    }
    bb.flip
    i
  }
}
