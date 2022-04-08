package lamp.nn

import org.scalatest.funsuite.AnyFunSuite
import org.saddle._
import lamp.autograd.{Variable, const}
import lamp.Sc
import lamp.Scope
import lamp.STen
import lamp.STenOptions
import lamp.saddle._

class LogisticSuite extends AnyFunSuite {

  def logisticRegression[S: Sc](dim: Int, k: Int, tOpt: STenOptions) =
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
        .toOption
        .get
      val target =
        lamp.saddle
          .fromLongMat(
            Mat(data.firstCol("label").toVec.map(_.toLong)),
            cuda
          )
          .squeeze

      val x =
        const(lamp.saddle.fromMat(data.filterIx(_ != "label").toMat, cuda))

      val model = logisticRegression(
        x.sizes(1).toInt,
        10,
        if (cuda) STenOptions.d.cudaIndex(0) else STenOptions.d
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
          val argm = output.value.argmax(1, false)
          val r = argm.toLongMat.toVec
          r
        }
        val correct = prediction.zipMap(data.firstCol("label").toVec)((a, b) =>
          if (a == b) 1d else 0d
        )
        val classWeights = STen.ones(List(10), x.options)
        val loss: Variable = output.nllLoss(target, classWeights)
        lastAccuracy = correct.mean2
        lastLoss = loss.value.toMat.raw(0)
        val gradients = model.gradients(loss)
        optim.step(gradients, 1d)
        i += 1
      }
      assert(lastAccuracy > 0.6)
      assert(lastLoss < 100d)
      ()
    }
  }
}
