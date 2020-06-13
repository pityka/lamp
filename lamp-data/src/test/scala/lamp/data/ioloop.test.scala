package lamp.data

import org.scalatest.funsuite.AnyFunSuite
import org.saddle._
import org.saddle.order._
import org.saddle.ops.BinOps._
import lamp.autograd.{Variable, const, param, TensorHelpers}
import lamp.nn._
import aten.ATen
import aten.TensorOptions

class IOLoopSuite extends AnyFunSuite {

  def logisticRegression(dim: Int, k: Int, tOpt: TensorOptions) =
    Sequential(
      Linear(dim, k, tOpt = tOpt),
      Fun(_.logSoftMax(dim = 1))
    )

  def test1(id: String)(fun: Boolean => Unit) = {
    test(id) { fun(false) }
    test(id + "/CUDA", CudaTest) { fun(true) }
  }

  test1("mnist tabular full batch") { cuda =>
    val device = if (cuda) CudaDevice(0) else CPU
    val data = org.saddle.csv.CsvParser
      .parseSourceWithHeader[Double](
        scala.io.Source
          .fromInputStream(getClass.getResourceAsStream("/mnist_test.csv"))
      )
      .right
      .get
    val x =
      TensorHelpers.fromMat(data.filterIx(_ != "label").toMat, cuda)
    val target = ATen.squeeze_0(
      TensorHelpers.fromLongMat(
        Mat(data.firstCol("label").toVec.map(_.toLong)),
        cuda
      )
    )
    val classWeights = ATen.ones(Array(10), x.options())

    val model = SupervisedModel[Unit](
      logisticRegression(data.numCols - 1, 10, device.options),
      (),
      LossFunctions.NLL(10, classWeights)
    )

    val trainedModel = IOLoops.epochs(
      model = model,
      optimizerFactory = SGDW
        .factory(
          learningRate = simple(0.0001),
          weightDecay = simple(0.001d)
        ),
      trainBatchesOverEpoch =
        () => BatchStream.fromFullBatch(x, target, device),
      validationBatchesOverEpoch =
        () => BatchStream.fromFullBatch(x, target, device),
      epochs = 50,
      trainingCallback = TrainingCallback.noop,
      validationCallback = ValidationCallback.noop,
      checkpointFile = None,
      checkpointFrequency = 1,
      minimumCheckpointFile = None
    )

    val (loss, output) = trainedModel
      .flatMap(_.lossAndOutput(x, target).allocated.map(_._1))
      .unsafeRunSync

    assert(loss < 900)

  }
  test1("mnist tabular mini batch") { cuda =>
    val device = if (cuda) CudaDevice(0) else CPU
    val data = org.saddle.csv.CsvParser
      .parseSourceWithHeader[Double](
        scala.io.Source
          .fromInputStream(getClass.getResourceAsStream("/mnist_test.csv"))
      )
      .right
      .get
    val x =
      TensorHelpers.fromMat(data.filterIx(_ != "label").toMat, cuda)
    val target = ATen.squeeze_0(
      TensorHelpers.fromLongMat(
        Mat(data.firstCol("label").toVec.map(_.toLong)),
        cuda
      )
    )
    val classWeights = ATen.ones(Array(10), x.options())

    val model = SupervisedModel(
      logisticRegression(data.numCols - 1, 10, device.options),
      (),
      LossFunctions.NLL(10, classWeights)
    )

    val trainedModel = IOLoops.epochs(
      model = model,
      optimizerFactory = SGDW
        .factory(
          learningRate = simple(0.0001),
          weightDecay = simple(0.001d)
        ),
      trainBatchesOverEpoch =
        () => BatchStream.minibatchesFromFull(200, true, x, target, device),
      validationBatchesOverEpoch =
        () => BatchStream.minibatchesFromFull(200, true, x, target, device),
      epochs = 50,
      trainingCallback = TrainingCallback.noop,
      validationCallback = ValidationCallback.noop,
      checkpointFile = None,
      checkpointFrequency = 1,
      minimumCheckpointFile = None
    )

    val (loss, output) = trainedModel
      .flatMap(_.lossAndOutput(x, target).allocated.map(_._1))
      .unsafeRunSync
    assert(loss < 900)

  }
}
