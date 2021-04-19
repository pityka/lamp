package lamp.data

import org.scalatest.funsuite.AnyFunSuite
import org.saddle._
import lamp.autograd.{const}
import lamp.nn._
import lamp.{CPU, CudaDevice}
import lamp.Scope
import lamp.STen
import lamp.STenOptions
import cats.effect.unsafe.implicits.global
import lamp.SinglePrecision

class MixedPrecisionSuite extends AnyFunSuite {
  implicit val graphconf = lamp.autograd.implicits.defaultGraphConfiguration
    .copy(downCastEnabled = true)
  def mlp(dim: Int, k: Int, tOpt: STenOptions)(implicit
      pool: Scope
  ) =
    sequence(
      MLP(dim, k, List(16, 16), tOpt, dropout = 0.2),
      Fun(implicit pool => _.logSoftMax(dim = 1))
    )

  def test1(id: String)(fun: Boolean => Unit) = {
    test(id) { fun(false) }
    test(id + "/CUDA", CudaTest) { fun(true) }
  }

  test1("mnist tabular mini batch") { cuda =>
    val stop = TensorLogger.start()(println _, (_, _) => true, 5000, 10000, 0)
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
        .row(0 -> 1000)
      val testDataTensor =
        STen.fromMat(testData.filterIx(_ != "label").toMat, cuda)
      val testTarget =
        STen
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
        .row(0 -> 1000)
      val trainDataTensor =
        STen.fromMat(trainData.filterIx(_ != "label").toMat, cuda)
      val trainTarget = STen
        .fromLongMat(
          Mat(trainData.firstCol("label").toVec.map(_.toLong)),
          cuda
        )
        .squeeze
      val classWeights = STen.ones(List(10), device.options(SinglePrecision))

      val model = SupervisedModel(
        mlp(784, 10, device.options(SinglePrecision)),
        LossFunctions.NLL(10, classWeights)
      )

      {
        val input = const(testDataTensor.castToFloat)
        val output = model.module.forward(input)
        val file = java.io.File.createTempFile("dfs", ".onnx")
        lamp.onnx.serializeToFile(
          file,
          output
        ) {
          case x if x == output =>
            lamp.onnx.VariableInfo(output, "output", input = false)
          case x if x == input =>
            lamp.onnx.VariableInfo(input, "node features", input = true)

        }
        println(file)

      }

      val rng = org.saddle.spire.random.rng.Cmwc5.apply()
      val makeValidationBatch = () =>
        BatchStream.minibatchesFromFull(
          200,
          true,
          testDataTensor.castToFloat,
          testTarget,
          rng
        )
      val makeTrainingBatch = () =>
        BatchStream.minibatchesFromFull(
          200,
          true,
          trainDataTensor.castToFloat,
          trainTarget,
          rng
        )

      val (_, trainedModel, _) = IOLoops
        .withSWA(
          model = model,
          optimizerFactory = SGDW
            .factory(
              learningRate = simple(0.01),
              weightDecay = simple(0.001d)
            ),
          trainBatchesOverEpoch = makeTrainingBatch,
          validationBatchesOverEpoch = Some(makeValidationBatch),
          warmupEpochs = 1,
          swaEpochs = 1,
          logger = Some(scribe.Logger("sdf"))
        )
        .unsafeRunSync()
      val acc = STen.scalarDouble(0d, testDataTensor.options)
      val (numExamples, _) = trainedModel
        .addTotalLossAndReturnGradientsAndNumExamples(
          const(testDataTensor.castToFloat),
          testTarget,
          acc
        )
      val loss = acc.toMat.raw(0) / numExamples
      assert(loss.isFinite)

    }
    stop.stop()
    TensorLogger.detailAllTensorOptions(println)
    assert(TensorLogger.queryActiveTensorOptions().size <= 3)
    assert(TensorLogger.queryActiveTensors().size == 0)
    ()
  }
}
