package lamp.nn

import org.scalatest.funsuite.AnyFunSuite
import org.saddle._
import lamp.autograd.{Variable, const}
import lamp.Sc
import lamp.Scope
import lamp.STen
import lamp.STenOptions
import lamp.saddle._
import lamp.autograd.Autograd.BackpropForwardCache
import lamp.autograd.Autograd.CacheStrategy
import aten.Tensor
import lamp.CPU
import lamp.MPS
import lamp.CudaDevice
import lamp.Device
import lamp.SinglePrecision

class LogisticSuite extends AnyFunSuite {
  
  def logisticRegression[S: Sc](dim: Int, k: Int, tOpt: STenOptions) =
    Sequential(
      Linear(dim, k, tOpt = tOpt),
      Fun(_ => input => input.logSoftMax(dim = 1))
    )

  def test1(id: String)(fun: Device => Unit) = {
    test(id) { fun(CPU) }
    if (Tensor.hasMps) {test(id+" MPS") { fun(MPS) }}
    if (Tensor.hasCuda) {test(id + "/CUDA" ) { fun(CudaDevice(0)) }}
  }

  test1("mnist tabular") { cuda =>
    Scope.root { implicit scope =>
      implicit val fw : BackpropForwardCache = BackpropForwardCache.empty(CacheStrategy.NeverCache,Nil)
      val data = org.saddle.csv.CsvParser
        .parseInputStreamWithHeader[Double](
          
              new java.util.zip.GZIPInputStream(
                getClass.getResourceAsStream("/mnist_test.csv.gz")
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
        const(lamp.saddle.fromMat(data.filterIx(_ != "label").toMat, cuda).castToFloat)

      val model = logisticRegression(
        x.eval.sizes(1).toInt,
        10,
        cuda.options(SinglePrecision)
      )

      val optim = SGDW(
        model.parameters.map(v => (v._1.constantValue, v._2)),
        learningRate = simple(0.0001),
        weightDecay = simple(0.001d)
      )

      var lastAccuracy = 0d
      var lastLoss = 1000000d
      var i = 0
      while (i < 300) {
        val output = model.forward(x)
        val prediction = {
          val argm = output.eval.argmax(1, false)
          val r = argm.toLongMat.toVec
          r
        }
        val correct = prediction.zipMap(data.firstCol("label").toVec)((a, b) =>
          if (a == b) 1d else 0d
        )
        val classWeights = STen.ones(List(10), x.options)
        val loss = output.nllLoss(target, classWeights).persist
        lastAccuracy = correct.mean2
        lastLoss = loss.forward.toMat.raw(0)
        val pd = ParameterGradients.empty
        val gradients = model.computeGradientsAndDestroyGraph(loss, pd.map.map(v => (v._1:Variable,v._2)), scope,true,fw,true)
        optim.step(model.parameters.map(v => gradients(v._1)), 1d)
        i += 1
      }
      assert(lastAccuracy > 0.6)
      assert(lastLoss < 100d)
      ()
    }
  }
}
