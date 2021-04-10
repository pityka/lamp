package lamp.tabular

import org.saddle._
import org.saddle.ops.BinOps._
import org.scalatest.funsuite.AnyFunSuite
import aten.Tensor
import lamp.SinglePrecision
import lamp.CPU
import lamp.CudaDevice
import lamp.Device
import lamp.DoublePrecision
import lamp.STen
import lamp.Scope
import cats.effect.unsafe.implicits.global

object TestTrain {
  def train(
      features: STen,
      target: STen,
      dataLayout: Seq[Metadata],
      targetType: TargetType,
      device: Device
  )(implicit scope: Scope) = {
    val precision =
      if (features.options.isDouble) DoublePrecision
      else if (features.options.isFloat) SinglePrecision
      else throw new RuntimeException("Expected float or double tensor")
    val numInstances = features.sizes.apply(0).toInt

    val minibatchSize = 1024
    val rng = org.saddle.spire.random.rng.Cmwc5.apply()
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
      logger = None,
      ensembleFolds = ensembleFolds,
      learningRate = 0.001,
      knnMinibatchSize = 512,
      rng = rng
    )(scope)
  }
}

class HousePricesSuite extends AnyFunSuite {

  test("regression") {
    Scope.root { implicit scope =>
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
        .toOption
        .get
      val rawTrainingData = rawTrainingData0.row(0 -> 999)
      val rawTestData = rawTrainingData0.row(1000 -> *)

      val trainTarget1 = rawTrainingData
        .firstCol("SalePrice")
        .toVec
        .map(_.toDouble)
        .map(math.log)

      val ecdf = ECDF(trainTarget1)

      val trainTarget =
        STen
          .fromMat(
            Mat(ecdf(trainTarget1).map(math.log)),
            CPU,
            SinglePrecision
          )
          .squeeze

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

      val predictedTest = {
        val ((trainingFeatures, metadata)) = StringMetadata
          .convertFrameToTensor(
            rawTrainingFeatures,
            preMeta.map(_._2),
            CPU,
            SinglePrecision,
            oneHotThreshold = oneHotThreshold
          )

        val ((testFeatures, metadata2)) = StringMetadata
          .convertFrameToTensor(
            rawTestFeatures,
            preMeta.map(_._2),
            CPU,
            SinglePrecision,
            oneHotThreshold = oneHotThreshold
          )
        assert(metadata == metadata2)
        for {
          trained <- train(
            trainingFeatures,
            trainTarget,
            metadata,
            ECDFRegression,
            device
          )
          _ = {
            // println("Validation losses: " + trained.validationLosses)
          }
          predicted <- trained.predict(testFeatures).map { modelOutput =>
            modelOutput.toMat

          }

        } yield {
          predicted.col(0)
        }

      }.unsafeRunSync()

      val error =
        math.sqrt(
          (ecdf.inverse(predictedTest.map(math.exp)) - testTarget
            .col(0)).map(v => v * v).mean
        )
      assert(error < 0.5)

    }
  }

}
