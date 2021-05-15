package lamp.tabular

import java.io.File
import java.io.FileOutputStream
import upickle.default._

import lamp.DoublePrecision
import lamp.SinglePrecision
import lamp.CPU
import lamp.extratrees.ClassificationTree
import lamp.extratrees.RegressionTree
import lamp.Scope
import lamp.STenOptions
import cats.effect.unsafe.implicits.global

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
    val src = scala.io.Source.fromFile(path)
    val json = src.mkString
    src.close
    val dto = read[DTO](json)

    val selectionModels = dto.selectionModels.map {

      case ExtratreesDto(trees) =>
        ExtratreesBase(trees)
      case KnnDto(k, file, _) =>
        val tensors = lamp.data.Reader.readTensorsFromFile(
          file = new File(file),
          device = CPU
        )
        KnnBase(
          k,
          tensors.head,
          tensors.drop(2),
          tensors(1)
        )
      case NNDto(hiddenSize, _, file, _) =>
        val tensors = lamp.data.Reader.readTensorsFromFile(
          file = new File(file),
          device = CPU
        )
        NNBase(hiddenSize, tensors)
    }
    val baseModels = dto.baseModels.map { case files =>
      files.map {
        case ExtratreesDto(trees) =>
          ExtratreesBase(trees)
        case NNDto(hiddenSize, _, file, _) =>
          val tensors = lamp.data.Reader.readTensorsFromFile(
            file = new File(file),
            device = CPU
          )
          NNBase(hiddenSize, tensors)
        case KnnDto(k, file, _) =>
          val t = lamp.data.Reader.readTensorsFromFile(
            file = new File(file),
            device = CPU
          )
          KnnBase(
            k,
            t.head,
            t.drop(2),
            t(1)
          )
      }

    }

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

        lamp.data.Writer
          .writeTensorsIntoFile(tensors, new File(path))
          .unsafeRunSync()
        NNDto(
          hiddenSize,
          tensors.size,
          path,
          tensors.map(t => Scope.leak { implicit scope => dataType(t.options) })
        )
      case (KnnBase(k, tensor1, tensors, tensor2), idx) =>
        val path = outPath + ".selection.knn." + idx

        lamp.data.Writer
          .writeTensorsIntoFile(
            (List(tensor1, tensor2) ++ tensors),
            new File(path)
          )
          .unsafeRunSync()
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
            lamp.data.Writer
              .writeTensorsIntoFile(
                (List(tensor1, tensor2) ++ tensors),
                new File(path)
              )
              .unsafeRunSync()
            KnnDto(
              k,
              path,
              (List(tensor1, tensor2) ++ tensors).map(t =>
                Scope.leak { implicit scope => dataType(t.options) }
              )
            )
          case (NNBase(hiddenSize, tensors), idx1) =>
            val path = outPath + ".base.nn." + idx0 + "." + idx1
            lamp.data.Writer
              .writeTensorsIntoFile(tensors, new File(path))
              .unsafeRunSync()

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
