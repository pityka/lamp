package lamp.nn

import org.scalatest.funsuite.AnyFunSuite
import org.saddle._
import lamp.autograd.{ const}
import lamp.Sc
import lamp.Scope
import lamp.STen
import lamp.STenOptions
import lamp.Device
import lamp.CPU
import lamp.MPS
import lamp.CudaDevice
import lamp.SinglePrecision

class SupervisedModelSuite extends AnyFunSuite {
  def logisticRegression[S: Sc](dim: Int, k: Int, tOpt: STenOptions) =
    Sequential(
      Linear(dim, k, tOpt = tOpt),
      Fun(_ => input => input.logSoftMax(dim = 1))
    )

  def test1(id: String)(fun: Device => Unit) = {
    test(id) { fun(CPU) }
    if (aten.Tensor.hasMps) {test(id+" MPS") { fun(MPS) }}
    if (aten.Tensor.hasCuda()) {test(id + "/CUDA") { fun(CudaDevice(0)) }}
  }

  test1("mnist tabular") { cuda =>
    Scope.root { implicit scope =>
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
      val supervised = SupervisedModel(
        module = model,
        lossFunction = LossFunctions.NLL(10,STen.ones(List(10), x.constantValue.options)),
        printMemoryAllocations = false
      )(TrainingMode.identity)
      

      var i = 0
      var lastLoss = 10000d
      while (i < 300) {
        val loss = STen.zeros(List(1),cuda.options(SinglePrecision))
        val (_,pd2) = supervised.addTotalLossAndReturnGradientsAndNumExamples(
          samples = x,
          target = target,
          acc = loss,
          zeroGrad = true,
          switchStream = false,
          pd = ParameterGradients.empty,
          
        )
        lastLoss = loss.toDevice(CPU).toDoubleArray(0) / x.constantValue.shape(0)
        optim.step(pd2.mapParameters(model.parameters.map(_._1)), 1d)
        i += 1
      }
      assert(lastLoss < 100)
      ()
    }
  }
}
