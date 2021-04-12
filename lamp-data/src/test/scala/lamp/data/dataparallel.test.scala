package lamp.data

import org.scalatest.funsuite.AnyFunSuite
import org.saddle._
import lamp.autograd.{const}
import lamp.nn._
import lamp.CPU
import lamp.DoublePrecision
import lamp.Scope
import lamp.STen
import lamp.STenOptions
import cats.effect.unsafe.implicits.global
import aten.Tensor
import lamp.CudaDevice

class DataParallelLoopSuite extends AnyFunSuite {
  def logisticRegression(dim: Int, k: Int, tOpt: STenOptions)(implicit
      pool: Scope
  ) =
    Seq2(
      Linear(dim, k, tOpt = tOpt),
      Fun(implicit pool => _.logSoftMax(dim = 1))
    )

  test("mnist tabular mini batch - data parallel gpu", CudaTest) {
    Scope.root { implicit scope =>
      Tensor.manual_seed(123L)
      val device = CPU
      val device2 = CudaDevice(0)
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
      val x =
        STen.fromMat(data.filterIx(_ != "label").toMat)
      val target =
        STen
          .fromLongMat(
            Mat(data.firstCol("label").toVec.map(_.toLong))
          )
          .squeeze

      val classWeights = STen.ones(List(10), x.options)

      val model1 = SupervisedModel(
        logisticRegression(
          data.numCols - 1,
          10,
          device.options(DoublePrecision)
        ),
        LossFunctions.NLL(10, classWeights)
      )
      val model2 = SupervisedModel(
        logisticRegression(
          data.numCols - 1,
          10,
          device2.options(DoublePrecision)
        ),
        LossFunctions.NLL(10, device2.to(classWeights))
      )

      val rng = org.saddle.spire.random.rng.Cmwc5.fromTime(2342L)

      val (_, trainedModel, _) = IOLoops
        .epochs(
          model = model1,
          optimizerFactory = SGDW
            .factory(
              learningRate = simple(0.0001),
              weightDecay = simple(0.001d)
            ),
          trainBatchesOverEpoch =
            () => BatchStream.minibatchesFromFull(200, true, x, target, rng),
          validationBatchesOverEpoch = Some(() =>
            BatchStream.minibatchesFromFull(200, true, x, target, rng)
          ),
          epochs = 50,
          trainingCallback = TrainingCallback.noop,
          validationCallback = ValidationCallback.noop,
          checkpointFile = None,
          minimumCheckpointFile = None,
          dataParallelModels = List(model2)
        )
        .unsafeRunSync()

      val acc = STen.scalarDouble(0d, x.options)
      val (n, _) = trainedModel
        .addTotalLossAndReturnGradientsAndNumExamples(const(x), target, acc)
      val loss = acc.toMat.raw(0) / n
      println(loss)
      assert(loss < 1.5)
    }
  }
}
