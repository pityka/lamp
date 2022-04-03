package lamp.data

import org.scalatest.funsuite.AnyFunSuite
import org.saddle._
import lamp.autograd.{const}
import lamp.nn._
import lamp.{CPU, CudaDevice, DoublePrecision}
import lamp.Scope
import lamp.STen
import lamp.STenOptions
import cats.effect.unsafe.implicits.global

class PerturbedLossSuite extends AnyFunSuite {
  def mlp(dim: Int, k: Int, tOpt: STenOptions)(implicit
      pool: Scope
  ) =
    sequence(
      MLP(dim, k, List(64,64, 32), tOpt, dropout = 0.2),
      Fun(implicit pool => _.logSoftMax(dim = 1))
    )

  def test1(id: String)(fun: Boolean => Unit) = {
    test(id) { fun(false) }
    test(id + "/CUDA", CudaTest) { fun(true) }
  }

  test1("perturbed loss - mnist") { cuda =>
    Scope.root { implicit scope =>
      val device = if (cuda) CudaDevice(0) else CPU
      val testData = org.saddle.csv.CsvParser
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
      val testDataTensor =
        lamp.saddle.fromMat(testData.filterIx(_ != "label").toMat, cuda)
      val testTarget =
        lamp.saddle
          .fromLongMat(
            Mat(testData.firstCol("label").toVec.map(_.toLong)),
            cuda
          )
          .squeeze

      val trainData = org.saddle.csv.CsvParser
        .parseSourceWithHeader[Double](
          scala.io.Source
            .fromInputStream(
              new java.util.zip.GZIPInputStream(
                getClass.getResourceAsStream("/mnist_train.csv.gz")
              )
            )
        )
        .toOption
        .get
      val trainDataTensor =
        lamp.saddle.fromMat(trainData.filterIx(_ != "label").toMat, cuda)
      val trainTarget = lamp.saddle
        .fromLongMat(
          Mat(trainData.firstCol("label").toVec.map(_.toLong)),
          cuda
        )
        .squeeze
      val classWeights = STen.ones(List(10), device.options(DoublePrecision))

      val model = SupervisedModel(
        mlp(784, 10, device.options(DoublePrecision)),
        LossFunctions.NLL(10, classWeights),
        // new SimpleLossCalculation
        new lamp.nn.PerturbedLossCalculation(0.01)
      )

      val rng = new scala.util.Random
      val makeValidationBatch = {
      (_: IOLoops.TrainingLoopContext) =>
        BatchStream.minibatchesFromFull(
          200,
          true,
          testDataTensor,
          testTarget,
          rng
        )}
      val makeTrainingBatch ={ (_: IOLoops.TrainingLoopContext) =>
        BatchStream.minibatchesFromFull(
          200,
          true,
          trainDataTensor,
          trainTarget,
          rng
        )
      }

        scribe.info("Start training loop")

      val (_, trainedModel, _, _, _) = IOLoops
        .epochs(
          model = model,
          optimizerFactory = SGDW
            .factory(
              learningRate = simple(0.01),
              weightDecay = simple(0.0d)
            ),
          trainBatchesOverEpoch = makeTrainingBatch,
          validationBatchesOverEpoch = Some(makeValidationBatch),
          epochs = 100,
          logger = Some(scribe.Logger("sdf"))
        )
        .unsafeRunSync()

      val acc = STen.scalarDouble(0d, testDataTensor.options)
      val (numExamples, _) = trainedModel
        .addTotalLossAndReturnGradientsAndNumExamples(
          const(testDataTensor),
          testTarget,
          acc,
          true
        )
      val loss = acc.toDoubleArray.head / numExamples
      println(loss)
      assert(loss < 0.25)

    }

    ()
  }

}
