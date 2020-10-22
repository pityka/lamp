package lamp.nn

import org.scalatest.funsuite.AnyFunSuite
import org.saddle._
import lamp.autograd.{Variable, const}
import aten.ATen
import aten.TensorOptions
import lamp.Sc
import lamp.Scope
import lamp.TensorHelpers
class LogisticSuite extends AnyFunSuite {

  def logisticRegression[S: Sc](dim: Int, k: Int, tOpt: TensorOptions) =
    Sequential(
      Linear(dim, k, tOpt = tOpt),
      Fun(scope => input => input.logSoftMax(dim = 1)(scope))
    )

  def test1(id: String)(fun: Boolean => Unit) = {
    test(id) { fun(false) }
    test(id + "/CUDA", CudaTest) { fun(true) }
  }

  test1("mnist tabular") { cuda =>
    Scope.root { implicit scope =>
      val data = org.saddle.csv.CsvParser
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
          val r = TensorHelpers.toLongMat(argm).toVec
          argm.release
          r
        }
        val correct = prediction.zipMap(data.firstCol("label").toVec)((a, b) =>
          if (a == b) 1d else 0d
        )
        val classWeights = ATen.ones(Array(10), x.options)
        val loss: Variable = output.nllLoss(target, 10, classWeights)
        lastAccuracy = correct.mean2
        lastLoss = TensorHelpers.toMat(loss.value).raw(0)
        val gradients = model.gradients(loss)
        optim.step(gradients)
        i += 1
      }
      assert(lastAccuracy > 0.6)
      assert(lastLoss < 100d)
      ()
    }
  }
}
