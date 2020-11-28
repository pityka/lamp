package lamp.data

import org.scalatest.funsuite.AnyFunSuite
import org.saddle._
import lamp.autograd.{const}
import lamp.nn._
import aten.TensorOptions
import lamp.CudaDevice
import lamp.CPU
import lamp.DoublePrecision
import lamp.Scope
import lamp.STen

class IOLoopSuite extends AnyFunSuite {
  def logisticRegression(dim: Int, k: Int, tOpt: TensorOptions)(
      implicit pool: Scope
  ) =
    Seq2(
      Linear(dim, k, tOpt = tOpt),
      Fun(implicit pool => _.logSoftMax(dim = 1))
    )

  def test1(id: String)(fun: Boolean => Unit) = {
    test(id) { fun(false) }
    test(id + "/CUDA", CudaTest) { fun(true) }
  }

  test1("mnist tabular full batch") { cuda =>
    Scope.root { implicit scope =>
      val device = if (cuda) CudaDevice(0) else CPU
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
      val x =
        STen.fromMat(data.filterIx(_ != "label").toMat, cuda)
      val target =
        STen
          .fromLongMat(
            Mat(data.firstCol("label").toVec.map(_.toLong)),
            cuda
          )
          .squeeze

      val classWeights = STen.ones(Array(10), x.options)

      val model = SupervisedModel(
        logisticRegression(
          data.numCols - 1,
          10,
          device.options(DoublePrecision)
        ),
        LossFunctions.NLL(10, classWeights)
      )

      val (epoch, trainedModel, learningCurve) = IOLoops
        .withSWA(
          model = model,
          optimizerFactory = SGDW
            .factory(
              learningRate = simple(0.0001),
              weightDecay = simple(0.001d)
            ),
          trainBatchesOverEpoch =
            () => BatchStream.fromFullBatch(x, target, device),
          validationBatchesOverEpoch =
            Some(() => BatchStream.fromFullBatch(x, target, device)),
          warmupEpochs = 50,
          swaEpochs = 20,
          trainingCallback = TrainingCallback.noop,
          validationCallback = ValidationCallback.noop,
          checkpointFile = None,
          minimumCheckpointFile = None,
          returnMinValidationLossModel = List(1, 25, 50)
        )
        .unsafeRunSync()

      val (loss, _, _) =
        trainedModel.lossAndOutput(const(x), target).allocated.unsafeRunSync._1

      assert(epoch == 25)
      println(loss)

      assert(learningCurve.size == 70)

      assert(loss < 50)
    }
  }
  test1("mnist tabular mini batch") { cuda =>
    Scope.root { implicit scope =>
      val device = if (cuda) CudaDevice(0) else CPU
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
      val x =
        STen.fromMat(data.filterIx(_ != "label").toMat, cuda)
      val target =
        STen
          .fromLongMat(
            Mat(data.firstCol("label").toVec.map(_.toLong)),
            cuda
          )
          .squeeze

      val classWeights = STen.ones(Array(10), x.options)

      val model = SupervisedModel(
        logisticRegression(
          data.numCols - 1,
          10,
          device.options(DoublePrecision)
        ),
        LossFunctions.NLL(10, classWeights)
      )

      val rng = org.saddle.spire.random.rng.Cmwc5.apply()

      val trainedModel = IOLoops.epochs(
        model = model,
        optimizerFactory = SGDW
          .factory(
            learningRate = simple(0.0001),
            weightDecay = simple(0.001d)
          ),
        trainBatchesOverEpoch = () =>
          BatchStream.minibatchesFromFull(200, true, x, target, device, rng),
        validationBatchesOverEpoch = Some(() =>
          BatchStream.minibatchesFromFull(200, true, x, target, device, rng)
        ),
        epochs = 50,
        trainingCallback = TrainingCallback.noop,
        validationCallback = ValidationCallback.noop,
        checkpointFile = None,
        minimumCheckpointFile = None,
        prefetchData = true
      )

      val (loss, _, _) = trainedModel
        .flatMap(_._2.lossAndOutput(const(x), target).allocated.map(_._1))
        .unsafeRunSync
      assert(loss < 50)
    }
  }
}
