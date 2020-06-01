package lamp.data

import org.scalatest.funsuite.AnyFunSuite
import org.saddle._
import org.saddle.order._
import org.saddle.ops.BinOps._
import lamp.autograd.{Variable, const, param, TensorHelpers}
import lamp.nn._
import aten.ATen
import aten.TensorOptions

class LogisticSuite extends AnyFunSuite {

  def logisticRegression(dim: Int, k: Int, tOpt: TensorOptions) =
    Sequential(
      Linear(dim, k, tOpt = tOpt),
      Fun(_.logSoftMax)
    )

  def test1(id: String)(fun: Boolean => Unit) = {
    test(id) { fun(false) }
    test(id + "/CUDA", CudaTest) { fun(true) }
  }

  test1("mnist tabular") { cuda =>
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

    val model = SupervisedModel(
      logisticRegression(data.numCols - 1, 10, device.options),
      LossFunctions.NLL(10)
    )

    TrainLoops.iteratorEpochs(
      model = model,
      optimizerFactory = SGDW
        .factory(
          learningRate = simple(0.0001),
          weightDecay = simple(0.001d)
        ),
      trainBatchesOverEpoch = () => {
        val xcl = device.to(x)
        val tcl = device.to(target)
        List((xcl, tcl)).iterator
      },
      validationBatchesOverEpoch = () => {
        val xcl = device.to(x)
        val tcl = device.to(target)
        List((xcl, tcl)).iterator
      },
      epochs = 100
    ) { (validationOutput, validationTarget, validationLoss) =>
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
    // println("Validation loss: " + validationLoss + " " + correct.mean2)
    }

  }
}
