package lamp.table

import java.io.File
import java.nio.channels.Channels
import cats.effect.IO
import java.io.FileOutputStream
import cats.effect.kernel.Resource
import lamp.data.Writer
import lamp.table.io.schemas.Schemas.ColumnDataTypeDescriptor
import java.nio.channels.ReadableByteChannel
import java.io.FileInputStream
import lamp.Device
import lamp.Scope

package object io {

  private[lamp] def readTableDescriptorFromChannel(
      channel: ReadableByteChannel
  ): schemas.Schemas.TableDescriptor = {
    val is = Channels.newInputStream(channel)
    com.github.plokhotnyuk.jsoniter_scala.core
      .readFromStream[schemas.Schemas.TableDescriptor](is)
  }

  private[lamp] def readTableDescriptorFromFile(
      file: File
  ): schemas.Schemas.TableDescriptor = {
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
        case schemas.Schemas.TextColumnType(maxLength, pad, vocabulary) =>
          Column(data, TextColumnType(maxLength, pad, vocabulary), None)
        case schemas.Schemas.DateTimeColumnType =>
          Column(data, DateTimeColumnType(), None)
        case schemas.Schemas.F32ColumnType => Column(data, F32ColumnType, None)
        case schemas.Schemas.I64ColumnType => Column(data, I64ColumnType, None)
        case schemas.Schemas.BooleanColumnType =>
          Column(data, BooleanColumnType(), None)
        case schemas.Schemas.F64ColumnType => Column(data, F64ColumnType, None)
      }
    }
    import org.saddle._
    Table(columns = columns.toVector, colNames = d.columnNames.toIndex)
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
                    case _: BooleanColumnType =>
                      schemas.Schemas.BooleanColumnType
                    case _: DateTimeColumnType =>
                      schemas.Schemas.DateTimeColumnType
                    case TextColumnType(maxLength, pad, vocabulary) =>
                      schemas.Schemas.TextColumnType(maxLength, pad, vocabulary)
                    case I64ColumnType => schemas.Schemas.I64ColumnType
                    case F32ColumnType => schemas.Schemas.F32ColumnType
                    case F64ColumnType => schemas.Schemas.F64ColumnType
                  })
                  .toList
              val tableDescriptor = schemas.Schemas.TableDescriptor(
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
