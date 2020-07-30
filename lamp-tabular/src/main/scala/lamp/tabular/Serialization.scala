package lamp.tabular

import java.io.File
import java.io.FileOutputStream
import upickle.default._

import lamp.DoublePrecision
import lamp.SinglePrecision
import lamp.CPU
import org.saddle.scalar.ScalarTagFloat
import org.saddle.scalar.ScalarTagDouble
import java.io.FileInputStream

object Serialization {

  case class DTO(
      selectionModels: Seq[(Int, Int, String)],
      baseModels: Seq[(Int, Seq[(Int, String)])],
      dataLayout: Seq[Metadata],
      targetType: TargetType,
      precision: String,
      validationLosses: Seq[Double]
  )
  object DTO {
    import upickle.default.{ReadWriter => RW, macroRW}
    implicit val rw: RW[DTO] = macroRW
  }
  def loadModel(path: String) = {
    val json = scala.io.Source.fromFile(path).mkString
    val dto = read[DTO](json)
    val st = dto.precision match {
      case "single" => ScalarTagFloat
      case "double" => ScalarTagDouble
    }
    val selectionModels = lamp.data.Reader.sequence(dto.selectionModels.map {
      case (n, num, file) =>
        val channel = new FileInputStream(new File(file)).getChannel()
        val tensors = lamp.data.Reader.readTensorsFromChannel(
          types = 0 until num map (_ => st),
          channel = channel,
          device = CPU
        )
        channel.close
        tensors.map(v => (n, v))
    })
    val baseModels = lamp.data.Reader.sequence(dto.baseModels.map {
      case (n, files) =>
        lamp.data.Reader
          .sequence(files.map {
            case (num, file) =>
              val channel = new FileInputStream(new File(file)).getChannel()
              val tensors = lamp.data.Reader.readTensorsFromChannel(
                types = 0 until num map (_ => st),
                channel = channel,
                device = CPU
              )
              channel.close
              tensors
          })
          .map(v => (n, v))
    })
    for {

      selectionModels <- selectionModels.right
      baseModels <- baseModels.right
    } yield {
      EnsembleModel(
        selectionModels,
        baseModels,
        dto.dataLayout,
        dto.targetType,
        dto.precision match {
          case "single" => SinglePrecision
          case "double" => DoublePrecision
        },
        dto.validationLosses
      )
    }
  }
  def saveModel(model: EnsembleModel, outPath: String) = {
    val selectionFiles = model.selectionModels.zipWithIndex.map {
      case ((n, tensors), idx) =>
        val path = outPath + ".selection." + idx
        val channel =
          new FileOutputStream(new File(path)).getChannel()
        lamp.data.Writer.writeTensorsIntoChannel(tensors, channel)
        channel.close
        (n, tensors.size, path)
    }
    val baseFiles = model.baseModels.zipWithIndex.map {
      case ((n, listsOfTensors), idx0) =>
        val paths = listsOfTensors.zipWithIndex.map {
          case (tensors, idx1) =>
            val path = outPath + ".base." + idx0 + "." + idx1
            val channel =
              new FileOutputStream(new File(path)).getChannel()
            lamp.data.Writer.writeTensorsIntoChannel(tensors, channel)
            channel.close
            (tensors.size, path)
        }
        (n, paths)
    }
    val dto = DTO(
      selectionFiles,
      baseFiles,
      model.dataLayout,
      model.targetType,
      model.precision match {
        case DoublePrecision => "double"
        case SinglePrecision => "single"
      },
      model.validationLosses
    )
    val json = write(dto)
    val os = new FileOutputStream(new File(outPath))
    os.write(json.getBytes("UTF-8"))
    os.close

  }
}
