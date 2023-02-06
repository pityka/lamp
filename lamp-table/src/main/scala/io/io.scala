package lamp.table

import java.io.File
import java.nio.channels.Channels
import cats.effect.IO
import java.io.FileOutputStream
import cats.effect.kernel.Resource
import lamp.data.Writer
import lamp.table.io.schemas.ColumnDataTypeDescriptor
import java.nio.channels.ReadableByteChannel
import java.io.FileInputStream
import lamp.Device
import lamp.Scope

package object io {

  private[lamp] def readTableDescriptorFromChannel(
      channel: ReadableByteChannel
  ): schemas.TableDescriptor = {
    val is = Channels.newInputStream(channel)
    com.github.plokhotnyuk.jsoniter_scala.core
      .readFromStream[schemas.TableDescriptor](is)
  }

  private[lamp] def readTableDescriptorFromFile(
      file: File
  ): schemas.TableDescriptor = {
    val fis = new FileInputStream(file)
    try {
      readTableDescriptorFromChannel(fis.getChannel)
    } finally {
      fis.close
    }
  }

  def readTableFromFile(
      file: File,
      device: Device = lamp.CPU,
      pin: Boolean = false
  )(implicit scope: Scope) = {
    val d = readTableDescriptorFromFile(file)
    val tensors =
      lamp.data.Reader.readTensorData(d.columnValues, file, device, pin)
    val columns = d.columnTypes.zip(tensors).map { case (tpe, data) =>
      tpe match {
        case schemas.TextColumnType(maxLength, pad, vocabulary) =>
          Column(data, TextColumnType(maxLength, pad, vocabulary), None)
        case schemas.DateTimeColumnType =>
          Column(data, DateTimeColumnType(), None)
        case schemas.F32ColumnType => Column(data, F32ColumnType, None)
        case schemas.I64ColumnType => Column(data, I64ColumnType, None)
        case schemas.BooleanColumnType =>
          Column(data, BooleanColumnType(), None)
        case schemas.F64ColumnType => Column(data, F64ColumnType, None)
      }
    }
    import org.saddle._
    Table(columns=columns.toVector, colNames = d.columnNames.toIndex)
  }

  def writeTableToFile(
      table: Table,
      file: File,
      bufferSize: Int = 16384
  ): IO[Either[String, Unit]] = {
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
          val tensors = table.columns.map(_.values)
          Writer
            .writeTensorDataAndMakeDescriptor(
              tensors = tensors,
              dataChannel = dataChannel,
              bufferSize = bufferSize,
              location = dataPath.getName,
              initialByteOffset = 0L
            )
            .map { tensorListDescriptor =>
              val colTypeDescriptors: List[ColumnDataTypeDescriptor] =
                table.columns
                  .map(_.tpe match {
                    case _: BooleanColumnType  => schemas.BooleanColumnType
                    case _: DateTimeColumnType => schemas.DateTimeColumnType
                    case TextColumnType(maxLength, pad, vocabulary) =>
                      schemas.TextColumnType(maxLength, pad, vocabulary)
                    case I64ColumnType => schemas.I64ColumnType
                    case F32ColumnType => schemas.F32ColumnType
                    case F64ColumnType => schemas.F64ColumnType
                  })
                  .toList
              val tableDescriptor = schemas.TableDescriptor(
                columnTypes = colTypeDescriptors,
                columnNames = table.colNames.toSeq.toList,
                columnValues = tensorListDescriptor
              )
              com.github.plokhotnyuk.jsoniter_scala.core.writeToStream(
                tableDescriptor,
                Channels.newOutputStream(descriptorChannel)
              )
            }
        }
      }
  }
}
