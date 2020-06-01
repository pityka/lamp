package lamp.nn

import org.scalatest.funsuite.AnyFunSuite
import org.saddle._
import org.saddle.order._
import org.saddle.ops.BinOps._
import lamp.autograd.{Variable, const, param, TensorHelpers}
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
    val data = org.saddle.csv.CsvParser
      .parseSourceWithHeader[Double](
        scala.io.Source
          .fromInputStream(getClass.getResourceAsStream("/mnist_test.csv"))
      )
      .right
      .get
    val target = ATen.squeeze_0(
      TensorHelpers.fromLongMat(
        Mat(data.firstCol("label").toVec.map(_.toLong)),
        cuda
      )
    )
    val x =
      const(TensorHelpers.fromMat(data.filterIx(_ != "label").toMat, cuda))

    val model = logisticRegression(
      x.sizes(1).toInt,
      10,
      if (cuda) TensorOptions.d.cuda else TensorOptions.d.cpu
    )

    val optim = SGDW(
      model.parameters.map(v => (v._1.value, v._2)),
      learningRate = simple(0.0001),
      weightDecay = simple(0.001d)
    )

    var lastAccuracy = 0d
    var lastLoss = 1000000d
    var i = 0
    while (i < 300) {
      val output = model.forward(x)
      val prediction = {
        val argm = ATen.argmax(output.value, 1, false)
        val r = TensorHelpers.toMatLong(argm).toVec
        argm.release
        r
      }
      val correct = prediction.zipMap(data.firstCol("label").toVec)((a, b) =>
        if (a == b) 1d else 0d
      )

      val loss: Variable = output.nllLoss(target, 10)
      lastAccuracy = correct.mean2
      lastLoss = TensorHelpers.toMat(loss.value).raw(0)
      val gradients = model.gradients(loss)
      optim.step(gradients)
      i += 1
    }
    assert(lastAccuracy > 0.6)
    assert(lastLoss < 100d)
  // println(data)
  // val target = dat

  }
}
