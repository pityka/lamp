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
import org.saddle.scalar.ScalarTagLong
import lamp.extratrees.ClassificationTree
import lamp.extratrees.RegressionTree
import lamp.STen
import lamp.Scope
import lamp.STenOptions

object Serialization {

  sealed trait BaseModelDTO
  object BaseModelDTO {
    import upickle.default.{ReadWriter => RW, macroRW}
    implicit val rw: RW[BaseModelDTO] = macroRW
  }
  case class ExtratreesDto(
      trees: Either[Seq[ClassificationTree], Seq[RegressionTree]]
  ) extends BaseModelDTO
  object ExtratreesDto {
    import upickle.default.{ReadWriter => RW, macroRW}
    implicit val rw: RW[ExtratreesDto] = macroRW
  }
  case class KnnDto(k: Int, path: String, dataTypes: Seq[String])
      extends BaseModelDTO
  object KnnDto {
    import upickle.default.{ReadWriter => RW, macroRW}
    implicit val rw: RW[KnnDto] = macroRW
  }
  case class NNDto(
      hiddenSize: Int,
      numTensors: Int,
      path: String,
      dataTypes: Seq[String]
  ) extends BaseModelDTO
  object NNDto {
    import upickle.default.{ReadWriter => RW, macroRW}
    implicit val rw: RW[NNDto] = macroRW
  }

  case class DTO(
      selectionModels: Seq[BaseModelDTO],
      baseModels: Seq[Seq[BaseModelDTO]],
      dataLayout: Seq[Metadata],
      precision: String,
      targetType: TargetType,
      validationLosses: Seq[Double]
  )
  object DTO {
    import upickle.default.{ReadWriter => RW, macroRW}
    implicit val rw: RW[DTO] = macroRW
  }
  def loadModel(path: String)(implicit scope: Scope) = {
    val json = scala.io.Source.fromFile(path).mkString
    val dto = read[DTO](json)
    def scalarTag(p: String) = p match {
      case "single" => ScalarTagFloat
      case "double" => ScalarTagDouble
      case "long"   => ScalarTagLong
    }
    val selectionModels = lamp.data.Reader.sequence(dto.selectionModels.map {

      case ExtratreesDto(trees) =>
        Right(ExtratreesBase(trees))
      case KnnDto(k, file, dataTypes) =>
        val channel = new FileInputStream(new File(file)).getChannel()
        val tensors = lamp.data.Reader.readTensorsFromChannel(
          types = dataTypes.map(scalarTag),
          channel = channel,
          device = CPU
        )
        channel.close
        tensors.map(v =>
          KnnBase(
            k,
            STen.owned(v.head),
            v.drop(2).map(STen.owned),
            STen.owned(v(1))
          )
        )
      case NNDto(hiddenSize, _, file, dataTypes) =>
        val channel = new FileInputStream(new File(file)).getChannel()
        val tensors = lamp.data.Reader.readTensorsFromChannel(
          types = dataTypes.map(scalarTag),
          channel = channel,
          device = CPU
        )
        channel.close
        tensors.map(v => NNBase(hiddenSize, v.map(STen.owned)))
    })
    val baseModels = lamp.data.Reader.sequence(dto.baseModels.map {
      case files =>
        lamp.data.Reader
          .sequence(files.map {
            case ExtratreesDto(trees) =>
              Right(ExtratreesBase(trees))
            case NNDto(hiddenSize, _, file, dataTypes) =>
              val channel = new FileInputStream(new File(file)).getChannel()
              val tensors = lamp.data.Reader.readTensorsFromChannel(
                types = dataTypes.map(scalarTag),
                channel = channel,
                device = CPU
              )
              channel.close
              tensors.map(t => NNBase(hiddenSize, t.map(STen.owned)))
            case KnnDto(k, file, dataTypes) =>
              val channel = new FileInputStream(new File(file)).getChannel()
              val tensors = lamp.data.Reader.readTensorsFromChannel(
                types = dataTypes.map(scalarTag),
                channel = channel,
                device = CPU
              )
              channel.close
              tensors.map(t =>
                KnnBase(
                  k,
                  STen.owned(t.head),
                  t.drop(2).map(STen.owned),
                  STen.owned(t(1))
                )
              )
          })

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
    def dataType(b: STenOptions) = b.scalarTypeByte match {
      case 6 => "single"
      case 7 => "double"
      case 4 => "long"
    }
    val selectionFiles = model.selectionModels.zipWithIndex.map {
      case (ExtratreesBase(trees), _) =>
        ExtratreesDto(trees)
      case (NNBase(hiddenSize, tensors), idx) =>
        val path = outPath + ".selection.nn." + idx
        val channel =
          new FileOutputStream(new File(path)).getChannel()
        lamp.data.Writer.writeTensorsIntoChannel(tensors.map(_.value), channel)
        channel.close
        NNDto(
          hiddenSize,
          tensors.size,
          path,
          tensors.map(t => Scope.leak { implicit scope => dataType(t.options) })
        )
      case (KnnBase(k, tensor1, tensors, tensor2), idx) =>
        val path = outPath + ".selection.knn." + idx
        val channel =
          new FileOutputStream(new File(path)).getChannel()
        lamp.data.Writer
          .writeTensorsIntoChannel(
            (List(tensor1, tensor2) ++ tensors).map(_.value),
            channel
          )
        channel.close
        KnnDto(
          k,
          path,
          (List(tensor1, tensor2) ++ tensors).map(t =>
            Scope.leak { implicit scope => dataType(t.options) }
          )
        )
    }
    val baseFiles = model.baseModels.zipWithIndex.map {
      case (listOfModels, idx0) =>
        val paths = listOfModels.zipWithIndex.map {
          case (ExtratreesBase(trees), _) =>
            ExtratreesDto(trees)
          case (KnnBase(k, tensor1, tensors, tensor2), idx1) =>
            val path = outPath + ".base.knn." + idx0 + "." + idx1
            val channel =
              new FileOutputStream(new File(path)).getChannel()
            lamp.data.Writer
              .writeTensorsIntoChannel(
                (List(tensor1, tensor2) ++ tensors).map(_.value),
                channel
              )
            channel.close
            KnnDto(
              k,
              path,
              (List(tensor1, tensor2) ++ tensors).map(t =>
                Scope.leak { implicit scope => dataType(t.options) }
              )
            )
          case (NNBase(hiddenSize, tensors), idx1) =>
            val path = outPath + ".base.nn." + idx0 + "." + idx1
            val channel =
              new FileOutputStream(new File(path)).getChannel()
            lamp.data.Writer
              .writeTensorsIntoChannel(tensors.map(_.value), channel)
            channel.close
            NNDto(
              hiddenSize,
              tensors.size,
              path,
              tensors.map(t =>
                Scope.leak { implicit scope => dataType(t.options) }
              )
            )
        }
        paths
    }
    val dto = DTO(
      selectionFiles,
      baseFiles,
      model.dataLayout,
      model.precision match {
        case DoublePrecision => "double"
        case SinglePrecision => "single"
      },
      model.targetType,
      model.validationLosses
    )
    val json = write(dto)
    val os = new FileOutputStream(new File(outPath))
    os.write(json.getBytes("UTF-8"))
    os.close

  }
}
