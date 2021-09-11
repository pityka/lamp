package lamp.data

import org.scalatest.funsuite.AnyFunSuite
import org.saddle._
import lamp.autograd.{const}
import lamp.nn._
import lamp.CudaDevice
import lamp.CPU
import lamp.DoublePrecision
import lamp.Scope
import lamp.STen
import lamp.STenOptions
import cats.effect.unsafe.implicits.global

class IOLoopSuite extends AnyFunSuite {
  def logisticRegression(dim: Int, k: Int, tOpt: STenOptions)(implicit
      pool: Scope
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
        .toOption
        .get

      import scala.concurrent.duration._
      val tensorLogger =
        TensorLogger.start(1.seconds)(
          s => scribe.info(s),
          (_, _) => true,
          5000,
          60000,
          1
        )
      val x =
        STen.fromMat(data.filterIx(_ != "label").toMat, cuda)
      val target =
        STen
          .fromLongMat(
            Mat(data.firstCol("label").toVec.map(_.toLong)),
            cuda
          )
          .squeeze

      val classWeights = STen.ones(List(10), x.options)

      val model = SupervisedModel(
        logisticRegression(
          data.numCols - 1,
          10,
          device.options(DoublePrecision)
        ),
        LossFunctions.NLL(10, classWeights),
        printMemoryAllocations = true
      )

      val (epoch, trainedModel, learningCurve, _) = IOLoops
        .withSWA(
          model = model,
          optimizerFactory = SGDW
            .factory(
              learningRate = simple(0.0001),
              weightDecay = simple(0.001d)
            ),
          trainBatchesOverEpoch = (_: IOLoops.TrainingLoopContext) =>
            BatchStream.fromFullBatch(x, target, device),
          validationBatchesOverEpoch = Some((_: IOLoops.TrainingLoopContext) =>
            BatchStream.fromFullBatch(x, target, device)
          ),
          warmupEpochs = 50,
          swaEpochs = 20,
          trainingCallback = TrainingCallback.noop,
          validationCallback = ValidationCallback.noop,
          returnMinValidationLossModel = List(1, 25, 50)
        )
        .unsafeRunSync()

      val acc = STen.scalarDouble(0d, x.options)
      val (n, _) =
        trainedModel
          .addTotalLossAndReturnGradientsAndNumExamples(
            const(x),
            target,
            acc,
            true
          )
      val loss = acc.toMat.raw(0) / n
      tensorLogger.cancel()

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
        .toOption
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

      val classWeights = STen.ones(List(10), x.options)

      val model = SupervisedModel(
        logisticRegression(
          data.numCols - 1,
          10,
          device.options(DoublePrecision)
        ),
        LossFunctions.NLL(10, classWeights)
      )

      val rng = org.saddle.spire.random.rng.Cmwc5.apply()

      val (_, trainedModel, _, _, _) = IOLoops
        .epochs(
          model = model,
          optimizerFactory = SGDW
            .factory(
              learningRate = simple(0.0001),
              weightDecay = simple(0.001d)
            ),
          trainBatchesOverEpoch = (_: IOLoops.TrainingLoopContext) =>
            BatchStream.minibatchesFromFull(200, true, x, target, rng),
          validationBatchesOverEpoch = Some((_: IOLoops.TrainingLoopContext) =>
            BatchStream.minibatchesFromFull(200, true, x, target, rng)
          ),
          epochs = 50,
          trainingCallback = TrainingCallback.noop,
          validationCallback = ValidationCallback.noop,
          prefetch = true,
          printOptimizerAllocations = true
        )
        .unsafeRunSync()

      val acc = STen.scalarDouble(0d, x.options)
      val (n, _) = trainedModel
        .addTotalLossAndReturnGradientsAndNumExamples(
          const(x),
          target,
          acc,
          true
        )
      val loss = acc.toMat.raw(0) / n
      assert(loss < 50)
    }
  }
  test1("mnist tabular mini batch gradient accum") { cuda =>
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
        .toOption
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

      val classWeights = STen.ones(List(10), x.options)

      val model = SupervisedModel(
        logisticRegression(
          data.numCols - 1,
          10,
          device.options(DoublePrecision)
        ),
        LossFunctions.NLL(10, classWeights)
      )

      val rng = org.saddle.spire.random.rng.Cmwc5.apply()

      val (_, trainedModel, _, _, _) = IOLoops
        .epochs(
          model = model,
          optimizerFactory = SGDW
            .factory(
              learningRate = simple(0.0001),
              weightDecay = simple(0.001d)
            ),
          trainBatchesOverEpoch = (_: IOLoops.TrainingLoopContext) =>
            BatchStream.minibatchesFromFull(200, true, x, target, rng),
          validationBatchesOverEpoch = Some((_: IOLoops.TrainingLoopContext) =>
            BatchStream.minibatchesFromFull(200, true, x, target, rng)
          ),
          epochs = 50,
          trainingCallback = TrainingCallback.noop,
          validationCallback = ValidationCallback.noop,
          prefetch = true,
          printOptimizerAllocations = true,
          accumulateGradientOverNBatches = 2
        )
        .unsafeRunSync()

      val acc = STen.scalarDouble(0d, x.options)
      val (n, _) = trainedModel
        .addTotalLossAndReturnGradientsAndNumExamples(
          const(x),
          target,
          acc,
          true
        )
      val loss = acc.toMat.raw(0) / n
      assert(loss < 50)
    }
  }
}
