package lamp.data

import org.scalatest.funsuite.AnyFunSuite
import org.saddle._
import org.saddle.order._
import org.saddle.ops.BinOps._
import lamp.autograd.{Variable, const, param, TensorHelpers}
import lamp.nn._
import lamp.{CPU, CudaDevice, DoublePrecision}
import aten.ATen
import aten.TensorOptions
import aten.Tensor

class MLPSuite extends AnyFunSuite {

  def mlp(dim: Int, k: Int, tOpt: TensorOptions) =
    Sequential(
      Linear(dim, 64, tOpt = tOpt),
      Fun(_.gelu),
      Dropout(0.2, true),
      Linear(64, k, tOpt = tOpt),
      Fun(_.logSoftMax(dim = 1))
    )

  def test1(id: String)(fun: Boolean => Unit) = {
    test(id) { fun(false) }
    test(id + "/CUDA", CudaTest) { fun(true) }
  }

  test1("mnist tabular mini batch") { cuda =>
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
      (),
      LossFunctions.NLL(10, classWeights)
    )

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

    val validationCallback = new ValidationCallback {
      def apply(
          validationOutput: Tensor,
          validationTarget: Tensor,
          validationLoss: Double,
          epochCount: Long
      ): Unit = {
        val prediction = {
          val t = ATen.argmax(validationOutput, 1, false)
          val r = TensorHelpers
            .toMatLong(t)
            .toVec
          t.release
          r
        }
        val correct = prediction.zipMap(
          TensorHelpers.toMatLong(validationTarget).toVec
        )((a, b) => if (a == b) 1d else 0d)
      }
    }

    val trainedModel = IOLoops.epochs(
      model = model,
      optimizerFactory = SGDW
        .factory(
          learningRate = simple(0.0001),
          weightDecay = simple(0.001d)
        ),
      trainBatchesOverEpoch = makeTrainingBatch,
      validationBatchesOverEpoch = makeValidationBatch,
      epochs = 10,
      trainingCallback = TrainingCallback.noop,
      validationCallback = validationCallback,
      checkpointFile = None,
      minimumCheckpointFile = None
    )
    val (loss, output, numExamples) = trainedModel
      .flatMap(_.lossAndOutput(testDataTensor, testTarget).allocated.map(_._1))
      .unsafeRunSync

    assert(loss < 50)

  }
}
