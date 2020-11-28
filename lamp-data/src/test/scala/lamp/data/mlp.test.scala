package lamp.data

import org.scalatest.funsuite.AnyFunSuite
import org.saddle._
import lamp.autograd.{const}
import lamp.nn._
import lamp.{CPU, CudaDevice, DoublePrecision}
import aten.TensorOptions
import lamp.Scope
import lamp.STen

class MLPSuite extends AnyFunSuite {
  def mlp(dim: Int, k: Int, tOpt: TensorOptions)(
      implicit pool: Scope
  ) =
    sequence(
      MLP(dim, k, List(64, 32), tOpt, dropout = 0.2),
      Fun(implicit pool => _.logSoftMax(dim = 1))
    )

  def test1(id: String)(fun: Boolean => Unit) = {
    test(id) { fun(false) }
    test(id + "/CUDA", CudaTest) { fun(true) }
  }

  test1("mnist tabular mini batch") { cuda =>
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
        .right
        .get
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
        .right
        .get
      val trainDataTensor =
        STen.fromMat(trainData.filterIx(_ != "label").toMat, cuda)
      val trainTarget = STen
        .fromLongMat(
          Mat(trainData.firstCol("label").toVec.map(_.toLong)),
          cuda
        )
        .squeeze
      val classWeights = STen.ones(Array(10), device.options(DoublePrecision))

      val model = SupervisedModel(
        mlp(784, 10, device.options(DoublePrecision)),
        LossFunctions.NLL(10, classWeights)
      )

      val rng = org.saddle.spire.random.rng.Cmwc5.apply()
      val makeValidationBatch = () =>
        BatchStream.minibatchesFromFull(
          200,
          true,
          testDataTensor,
          testTarget,
          device,
          rng
        )
      val makeTrainingBatch = () =>
        BatchStream.minibatchesFromFull(
          200,
          true,
          trainDataTensor,
          trainTarget,
          device,
          rng
        )

      val (_, trainedModel, _) = IOLoops
        .withSWA(
          model = model,
          optimizerFactory = SGDW
            .factory(
              learningRate = simple(0.0001),
              weightDecay = simple(0.001d)
            ),
          trainBatchesOverEpoch = makeTrainingBatch,
          validationBatchesOverEpoch = Some(makeValidationBatch),
          warmupEpochs = 10,
          swaEpochs = 10
        )
        .unsafeRunSync()
      val (loss, _, _) = trainedModel
        .lossAndOutput(const(testDataTensor), testTarget)
        .allocated
        .map(_._1)
        .unsafeRunSync
      assert(loss < 3)

      {
        val input = const(testDataTensor)
        val output = trainedModel.module.forward(input)
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
    }
  }
}
