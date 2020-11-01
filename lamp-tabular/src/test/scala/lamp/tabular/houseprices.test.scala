package lamp.tabular

import org.saddle._
import org.saddle.ops.BinOps._
import org.scalatest.funsuite.AnyFunSuite
import aten.ATen
import lamp.TensorHelpers
import lamp.util.syntax
import aten.Tensor
import cats.effect.IO
import lamp.SinglePrecision
import lamp.CPU
import lamp.CudaDevice
import lamp.Device
import lamp.DoublePrecision
import lamp.StringMetadata

object TestTrain {
  def train(
      features: Tensor,
      target: Tensor,
      dataLayout: Seq[Metadata],
      targetType: TargetType,
      device: Device,
      logFrequency: Int
  ) = {
    val precision =
      if (features.options.isDouble) DoublePrecision
      else if (features.options.isFloat) SinglePrecision
      else throw new RuntimeException("Expected float or double tensor")
    val numInstances = features.sizes.apply(0).toInt

    val minibatchSize = 1024
    val rng = org.saddle.spire.random.rng.Cmwc5.apply
    val cvFolds =
      AutoLoop.makeCVFolds(
        numInstances,
        k = 4,
        2,
        rng
      )

    val ensembleFolds =
      AutoLoop
        .makeCVFolds(numInstances, k = 4, 2, rng)
    println("BBB")
    AutoLoop.train(
      dataFullbatch = features,
      targetFullbatch = target,
      folds = cvFolds,
      targetType = targetType,
      dataLayout = dataLayout,
      epochs = Seq(4, 8, 16),
      weighDecays = Seq(0.0001),
      dropouts = Seq(0.05),
      hiddenSizes = Seq(32),
      knnK = Seq(5),
      extratreesK = Seq(30),
      extratreesM = Seq(5),
      extratreesNMin = Seq(2),
      extratreeParallelism = 1,
      device = device,
      precision = precision,
      minibatchSize = minibatchSize,
      logFrequency = logFrequency,
      logger = None,
      ensembleFolds = ensembleFolds,
      learningRate = 0.001,
      prescreenHyperparameters = true,
      knnMinibatchSize = 512,
      rng = rng
    )
  }
}

class HousePricesSuite extends AnyFunSuite {

  test("regression") {
    import TestTrain.train
    val device = if (Tensor.cudnnAvailable()) CudaDevice(0) else CPU

    val rawTrainingData0 = org.saddle.csv.CsvParser
      .parseSourceWithHeader[String](
        scala.io.Source
          .fromInputStream(
            getClass.getResourceAsStream("/train.csv")
          ),
        recordSeparator = "\n"
      )
      .right
      .get
    val rawTrainingData = rawTrainingData0.row(0 -> 999)
    val rawTestData = rawTrainingData0.row(1000 -> *)

    val trainTarget1 = rawTrainingData
      .firstCol("SalePrice")
      .toVec
      .map(_.toDouble)
      .map(math.log)

    val ecdf = ECDF(trainTarget1)

    val trainTarget = ATen.squeeze_0(
      TensorHelpers.fromMat(
        Mat(ecdf(trainTarget1).map(math.log)),
        CPU,
        SinglePrecision
      )
    )
    val testTarget =
      Mat(
        rawTestData.firstCol("SalePrice").toVec.map(_.toDouble).map(math.log)
      )

    val rawTrainingFeatures =
      rawTrainingData.filterIx(ix => !Set("SalePrice", "Id").contains(ix))
    val rawTestFeatures =
      rawTestData.filterIx(ix => !Set("SalePrice", "Id").contains(ix))

    val preMeta = StringMetadata.inferMetaFromFrame(rawTrainingFeatures)

    val oneHotThreshold = 4

    val predictedTest = StringMetadata
      .convertFrameToTensor(
        rawTrainingFeatures,
        preMeta.map(_._2),
        CPU,
        SinglePrecision,
        oneHotThreshold = oneHotThreshold
      )
      .use {
        case ((trainingFeatures, metadata)) =>
          StringMetadata
            .convertFrameToTensor(
              rawTestFeatures,
              preMeta.map(_._2),
              CPU,
              SinglePrecision,
              oneHotThreshold = oneHotThreshold
            )
            .use {
              case ((testFeatures, metadata2)) =>
                assert(metadata == metadata2)
                println("AA")
                for {
                  trained <- train(
                    trainingFeatures,
                    trainTarget,
                    metadata,
                    ECDFRegression,
                    device,
                    logFrequency = 10
                  )
                  _ = {
                    // println("Validation losses: " + trained.validationLosses)
                  }
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

    val error =
      math.sqrt(
        (ecdf.inverse(predictedTest.map(math.exp)) - testTarget
          .col(0)).map(v => v * v).mean
      )
    assert(error < 0.5)

  }

}
