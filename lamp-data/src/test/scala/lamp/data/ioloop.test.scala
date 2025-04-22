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
import aten.Tensor
class IOLoopSuite extends AnyFunSuite {
  def logisticRegression(dim: Int, k: Int, tOpt: STenOptions)(implicit
      pool: Scope
  ) =
    Seq2(
      Linear(dim, k, tOpt = tOpt),
      Fun(_ => _.logSoftMax(dim = 1))
    )

  def test1(id: String)(fun: Boolean => Unit) = {
    test(id) { fun(false) }
    if (Tensor.hasCuda) {test(id + "/CUDA") { fun(true) }}
  }

  test1("mnist tabular full batch") { cuda =>
    Scope.root { implicit scope =>

      val device = if (cuda) CudaDevice(0) else CPU
      val data = org.saddle.csv.CsvParser
        .parseInputStreamWithHeader[Double](
          new java.util.zip.GZIPInputStream(
            getClass.getResourceAsStream("/mnist_test.csv.gz")
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
        lamp.saddle.fromMat(data.filterIx(_ != "label").toMat, cuda)
      val target =
        lamp.saddle
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
        printMemoryAllocations = false
      )

      val (epoch, trainedModel, learningCurve,_, _) = IOLoops
        .epochs(
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
          epochs = 100,
          trainingCallback = None,
          validationCallback = None,
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
            true,
            true,
            ParameterGradients.empty
          )
      val loss = acc.toDoubleArray.head / n
      tensorLogger.cancel()

      assert(epoch == 50)
      println(loss)

      assert(learningCurve.size == 100)

      assert(loss < 50)
      ()
    }
  }
  test1("mnist tabular mini batch") { cuda =>
    Scope.root { implicit scope =>

      val device = if (cuda) CudaDevice(0) else CPU
      val data = org.saddle.csv.CsvParser
        .parseInputStreamWithHeader[Double](
          new java.util.zip.GZIPInputStream(
            getClass.getResourceAsStream("/mnist_test.csv.gz")
          )
        )
        .toOption
        .get
      val x =
        lamp.saddle.fromMat(data.filterIx(_ != "label").toMat, cuda)
      val target =
        lamp.saddle
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

      val rng = new scala.util.Random()

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
          trainingCallback = None,
          validationCallback = None,
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
          true,
          true,
          ParameterGradients.empty
        )
      val loss = acc.toDoubleArray.head / n
      assert(loss < 50)
      ()
    }
  }
  test1("mnist tabular mini batch gradient accum") { cuda =>
    Scope.root { implicit scope =>
      val device = if (cuda) CudaDevice(0) else CPU
      val data = org.saddle.csv.CsvParser
        .parseInputStreamWithHeader[Double](
          new java.util.zip.GZIPInputStream(
            getClass.getResourceAsStream("/mnist_test.csv.gz")
          )
        )
        .toOption
        .get
      val x =
        lamp.saddle.fromMat(data.filterIx(_ != "label").toMat, cuda)
      val target =
        lamp.saddle
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

      val rng = new scala.util.Random()

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
          epochs = 100,
          trainingCallback = None,
          validationCallback = None,
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
          true,
          true,
          ParameterGradients.empty
        )
      val loss = acc.toDoubleArray.head / n
      assert(loss < 50)
      ()
    }
  }
}
