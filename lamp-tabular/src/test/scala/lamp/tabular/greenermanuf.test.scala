package lamp.tabular

import org.saddle._
import org.saddle.order._
import org.saddle.ops.BinOps._
import org.scalatest.funsuite.AnyFunSuite
import aten.ATen
import lamp.autograd._
import aten.TensorOptions
import org.scalatest.Tag
import lamp.syntax
import lamp.util.NDArray
import aten.Tensor
import cats.effect.IO
import cats.effect.concurrent.Ref
import lamp.nn._
import lamp.SinglePrecision
import lamp.CPU
import lamp.CudaDevice
import lamp.Device
import lamp.DoublePrecision
import scribe.Logger
import lamp.StringMetadata
import java.io.FileInputStream
import java.io.File

class GreenerManufacturingSuite extends AnyFunSuite {
  val cpuPool = new AllocatedVariablePool
  val cudaPool = new AllocatedVariablePool
  def selectPool(cuda: Boolean) = if (cuda) cudaPool else cpuPool

  ignore("regression") {

    def train(
        features: Tensor,
        target: Tensor,
        dataLayout: Seq[Metadata],
        targetType: TargetType,
        device: Device,
        logger: Option[Logger],
        logFrequency: Int
    ) = {
      implicit val pool = new AllocatedVariablePool
      val precision =
        if (features.options.isDouble) DoublePrecision
        else if (features.options.isFloat) SinglePrecision
        else throw new RuntimeException("Expected float or double tensor")
      val numInstances = features.sizes.apply(0).toInt

      val minibatchSize = 512
      val cvFolds =
        AutoLoop.makeCVFolds(
          numInstances,
          k = 2,
          1
        )

      val ensembleFolds =
        AutoLoop
          .makeCVFolds(numInstances, k = 2, 1)
      AutoLoop.train(
        dataFullbatch = features,
        targetFullbatch = target,
        folds = cvFolds,
        targetType = targetType,
        dataLayout = dataLayout,
        epochs = Seq(4, 8, 16, 32, 64, 128, 256, 512),
        weighDecays = Seq(0.0001, 0.001),
        dropouts = Seq(0.1, 0.95),
        hiddenSizes = Seq(64),
        device = device,
        precision = precision,
        minibatchSize = minibatchSize,
        logFrequency = logFrequency,
        logger = logger,
        ensembleFolds = ensembleFolds,
        learningRate = 0.001
      )
    }

    implicit val pool = new AllocatedVariablePool
    val device = if (Tensor.cudnnAvailable()) CudaDevice(0) else CPU
    val rawTrainingData0 = org.saddle.csv.CsvParser
      .parseSourceWithHeader[String](
        scala.io.Source
          .fromFile(
            "../test_data/kaggle/mercedes-benz-greener-manufacturing/train.csv"
          ),
        recordSeparator = "\n"
      )
      .right
      .get
    val rawTrainingData = rawTrainingData0 //.row(0 -> 3999)
    // val rawTestData = rawTrainingData0.row(4000 -> *)
    val rawTestData = org.saddle.csv.CsvParser
      .parseSourceWithHeader[String](
        scala.io.Source
          .fromFile(
            "../test_data/kaggle/mercedes-benz-greener-manufacturing/test.csv"
          ),
        recordSeparator = "\n"
      )
      .right
      .get

    println(rawTrainingData)

    val trainTarget1 = rawTrainingData
      .firstCol("y")
      .toVec
      .map(_.toDouble)

    val ecdf = ECDF(trainTarget1)

    println(ecdf)

    val trainTarget = ATen.squeeze_0(
      TensorHelpers.fromMat(
        Mat(ecdf(trainTarget1).map(math.log)),
        device,
        SinglePrecision
      )
    )
    val testTarget =
      Mat(
        rawTestData.firstCol("y").toVec.map(_.toDouble)
      )

    val rawTrainingFeatures =
      rawTrainingData.filterIx(ix => !Set("y", "ID").contains(ix))
    val rawTestFeatures =
      rawTestData.filterIx(ix => !Set("y", "ID").contains(ix))

    val preMeta = StringMetadata.inferMetaFromFrame(rawTrainingFeatures)

    println(preMeta.mkString("\n"))

    val oneHotThreshold = 4

    val predictedTest = StringMetadata
      .convertFrameToTensor(
        rawTrainingFeatures,
        preMeta.map(_._2),
        CPU,
        SinglePrecision,
        oneHotThreshold
      )
      .use {
        case ((trainingFeatures, metadata)) =>
          println(metadata.mkString("\n"))
          StringMetadata
            .convertFrameToTensor(
              rawTestFeatures,
              preMeta.map(_._2),
              CPU,
              SinglePrecision,
              oneHotThreshold
            )
            .use {
              case ((testFeatures, metadata2)) =>
                assert(metadata == metadata2)
                for {
                  trained <- train(
                    trainingFeatures,
                    trainTarget,
                    metadata,
                    ECDFRegression,
                    device,
                    Some(scribe.Logger("test")),
                    logFrequency = 10
                  )
                  predicted <- trained.predict(testFeatures).use {
                    modelOutput =>
                      IO {
                        modelOutput.toMat
                      }

                  }

                } yield {
                  predicted.col(0)
                }
            }
      }
      .unsafeRunSync()

    println(
      org.saddle.csv.CsvWriter
        .writeFrameToFile(
          Frame(
            "ID" -> Series(rawTestData.firstCol("ID").toVec),
            "y" -> Series(
              ecdf.inverse(predictedTest.map(math.exp)).map(_.toString)
            )
          ),
          "predicted_mf.csv",
          withRowIx = false
        )
    )
    // println(Mat(predictedTest, testTarget.col(0)))
    // println(math.pow(predictedTest.pearson(testTarget.col(0)), 2d))

  }

}
