package lamp.data

import org.scalatest.funsuite.AnyFunSuite
import org.saddle._
import lamp.autograd.{const, TensorHelpers}
import lamp.nn._
import lamp.{CPU, CudaDevice, DoublePrecision}
import aten.ATen
import aten.TensorOptions
import scribe.Level
import lamp.autograd.AllocatedVariablePool
import lamp.autograd.Mean

class MLPSuite extends AnyFunSuite {

  def mlp(dim: Int, k: Int, tOpt: TensorOptions)(
      implicit pool: AllocatedVariablePool
  ) =
    sequence(
      MLP(dim, k, List(64, 32), tOpt, dropout = 0.2),
      Fun(_.logSoftMax(dim = 1))
    )

  def test1(id: String)(fun: Boolean => Unit) = {
    test(id, SlowTest) { fun(false) }
    test(id + "/CUDA", CudaTest) { fun(true) }
  }

  test1("mnist tabular mini batch") { cuda =>
    implicit val pool = new AllocatedVariablePool
    val device = if (cuda) CudaDevice(0) else CPU
    val testData = org.saddle.csv.CsvParser
      .parseSourceWithHeader[Double](
        scala.io.Source
          .fromInputStream(
            new java.util.zip.GZIPInputStream(
              getClass.getResourceAsStream("/mnist_test.csv.gz")
            )
          )
      )
      .right
      .get
    val testDataTensor =
      TensorHelpers.fromMat(testData.filterIx(_ != "label").toMat, cuda)
    val testTarget = ATen.squeeze_0(
      TensorHelpers.fromLongMat(
        Mat(testData.firstCol("label").toVec.map(_.toLong)),
        cuda
      )
    )

    val trainData = org.saddle.csv.CsvParser
      .parseSourceWithHeader[Double](
        scala.io.Source
          .fromInputStream(
            new java.util.zip.GZIPInputStream(
              getClass.getResourceAsStream("/mnist_train.csv.gz")
            )
          )
      )
      .right
      .get
    val trainDataTensor =
      TensorHelpers.fromMat(trainData.filterIx(_ != "label").toMat, cuda)
    val trainTarget = ATen.squeeze_0(
      TensorHelpers.fromLongMat(
        Mat(trainData.firstCol("label").toVec.map(_.toLong)),
        cuda
      )
    )
    val classWeights = ATen.ones(Array(10), device.options(DoublePrecision))

    val model = SupervisedModel(
      mlp(784, 10, device.options(DoublePrecision)),
      LossFunctions.NLL(10, classWeights),
      InputGradientRegularizer(h = 0.01, lambda = 0.001)
    )

    assert(model.module.state.size == 18)

    val makeValidationBatch = () =>
      BatchStream.minibatchesFromFull(
        200,
        true,
        testDataTensor,
        testTarget,
        device
      )
    val makeTrainingBatch = () =>
      BatchStream.minibatchesFromFull(
        200,
        true,
        trainDataTensor,
        trainTarget,
        device
      )

    val logger = scribe
      .Logger("test")
      .clearHandlers()
      .clearModifiers()
      .withMinimumLevel(Level.Error)
    val validationCallback =
      ValidationCallback.logAccuracy(logger)

    val trainedModel = IOLoops
      .epochs(
        model = model,
        optimizerFactory = AdamW
          .factory(
            learningRate = simple(0.001),
            weightDecay = simple(0.0001d)
          ),
        trainBatchesOverEpoch = makeTrainingBatch,
        validationBatchesOverEpoch = Some(makeValidationBatch),
        epochs = 10,
        validationCallback = validationCallback
        // logger = Some(logger)
      )
      .unsafeRunSync()
    val (_, output, _) = trainedModel.asEval
      .lossAndOutput(const(testDataTensor), testTarget)
      .allocated
      .map(_._1)
      .unsafeRunSync()

    val loss =
      LossFunctions
        .NLL(10, classWeights)(const(output), testTarget, Mean)
        ._1
        .toMat
        .raw(0)

    val prediction = {
      val t = ATen.argmax(output, 1, false)
      val r = TensorHelpers
        .toLongMat(t)
        .toVec
      t.release
      r
    }
    val corrects = prediction.zipMap(
      TensorHelpers.toLongMat(testTarget).toVec
    )((a, b) => if (a == b) 1d else 0d)
    assert(corrects.mean2 > 0.93)
    assert(loss < 2.0)
  }
}
