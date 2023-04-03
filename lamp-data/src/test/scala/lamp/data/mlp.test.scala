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
import cats.effect.IO

class MLPSuite extends AnyFunSuite {
  def mlp(dim: Int, k: Int, tOpt: STenOptions)(implicit
      pool: Scope
  ) =
    sequence(
      MLP(dim, k, List(64, 32), tOpt, dropout = 0.2, numHeads=2),
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
        AdversarialTraining(2d)
      )

      val rng = new scala.util.Random
      val makeValidationBatch = (_: IOLoops.TrainingLoopContext) =>
        BatchStream.minibatchesFromFull(
          200,
          true,
          testDataTensor,
          testTarget,
          rng
        )
      val makeTrainingBatch = (_: IOLoops.TrainingLoopContext) =>
        BatchStream.minibatchesFromFull(
          200,
          true,
          trainDataTensor,
          trainTarget,
          rng
        )

      val stateFile =
        java.io.File.createTempFile("sdfs", "dsfsd").getAbsoluteFile()

      val (_, trainedModel, _, _) = IOLoops
        .withSWA(
          model = model,
          optimizerFactory = Shampoo
            .factory(
              learningRate = simple(0.01),
              momentum = simple(0.9)
              // weightDecay = simple(0.001d)
            ),
          trainBatchesOverEpoch = makeTrainingBatch,
          validationBatchesOverEpoch = Some(makeValidationBatch),
          warmupEpochs = 10,
          swaEpochs = 10,
          logger = Some(scribe.Logger("sdf")),
          checkpointState = Some((state: LoopState, _: Either[_, _]) =>
            IO {

              StateIO.writeToFile(stateFile, state)
              val state2 = StateIO.readFromFile(stateFile, device)
              assertLoopState(state, state2)
            }
          )
        )
        .unsafeRunSync()

      val state3 = StateIO
        .readFromFile(stateFile, device)
        .asInstanceOf[SimpleThenSWALoopState]

      println("run from checkpoint")
      IOLoops
        .withSWA(
          model = model,
          optimizerFactory = SGDW
            .factory(
              learningRate = simple(0.01),
              weightDecay = simple(0.001d)
            ),
          trainBatchesOverEpoch = makeTrainingBatch,
          validationBatchesOverEpoch = Some(makeValidationBatch),
          warmupEpochs = 10,
          swaEpochs = 10,
          logger = Some(scribe.Logger("sdf")),
          checkpointState = Some((state: LoopState, _: Either[_, _]) =>
            IO {

              StateIO.writeToFile(stateFile, state)
              val state2 = StateIO.readFromFile(stateFile, device)
              assertLoopState(state, state2)
            }
          ),
          initState = Some(state3)
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
      assert(loss < 0.8)

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
    stop.stop()
    TensorLogger.detailAllTensorOptions(println)
    assert(TensorLogger.queryActiveTensorOptions().size <= 3)
    println("Remaining:")
    TensorLogger.queryActiveTensors().foreach { td =>
      println(td.getShape())
      println(td.getStackTrace().mkString("\n"))
    }
    assert(TensorLogger.queryActiveTensors().size == 0)
    ()
  }

  def assertLoopState(state: LoopState, state2: LoopState): Unit = state match {
    case SimpleThenSWALoopState(simple, swa) =>
      val simple2 = state2.asInstanceOf[SimpleThenSWALoopState].simple
      val swa2 = state2.asInstanceOf[SimpleThenSWALoopState].swa

      assert(swa.isDefined == swa.isDefined)
      assertLoopState(simple, simple2)

      if (swa.isDefined) assertLoopState(swa.get, swa2.get)
    case SimpleLoopState(
          model,
          optimizer,
          epoch,
          lastValidationLoss,
          minValidationLoss,
          minValidationLossModel,
          learningCurve
        ) =>
      val st = state2.asInstanceOf[SimpleLoopState]
      assert(model.map(_.shape) == st.model.map(_.shape))
      assert(model.zip(st.model).forall { case (a, b) =>
        a.equalDeep(b)
      })
      assert(epoch == st.epoch)
      assert(learningCurve == st.learningCurve)
      assert(minValidationLoss == st.minValidationLoss)
      assert(lastValidationLoss == st.lastValidationLoss)
      assert(optimizer.map(_.shape) == st.optimizer.map(_.shape))
      assert(optimizer.zip(st.optimizer).forall { case (a, b) =>
        a.equalDeep(b)
      })
      assert(
        minValidationLossModel.map(_._1) == st.minValidationLossModel.map(_._1)
      )
      assert(
        minValidationLossModel.map(
          _._2.map(_.sizes.toList)
        ) == st.minValidationLossModel.map(_._2.map(_.sizes.toList))
      )
      assert(
        minValidationLossModel.toSeq
          .flatMap(_._2)
          .zip(st.minValidationLossModel.toSeq.flatMap(_._2))
          .forall { case (a, b) => aten.ATen.equal(a, b) }
      )
    case SWALoopState(
          model,
          optimizer,
          epoch,
          lastValidationLoss,
          minValidationLoss,
          numberOfAveragedModels,
          averagedModels,
          learningCurve
        ) =>
      val st = state2.asInstanceOf[SWALoopState]
      assert(model.map(_.shape) == st.model.map(_.shape))
      assert(model.zip(st.model).forall { case (a, b) =>
        a.equalDeep(b)
      })
      assert(epoch == st.epoch)
      assert(learningCurve == st.learningCurve)
      assert(minValidationLoss == st.minValidationLoss)
      assert(numberOfAveragedModels == st.numberOfAveragedModels)
      assert(lastValidationLoss == st.lastValidationLoss)
      assert(optimizer.map(_.shape) == st.optimizer.map(_.shape))
      assert(optimizer.zip(st.optimizer).forall { case (a, b) =>
        a.equalDeep(b)
      })
      assert(
        averagedModels.map(_.map(_.sizes().toList)) == st.averagedModels.map(
          _.map(_.sizes.toList)
        )
      )
      if (
        !averagedModels.toSeq.flatten
          .zip(st.averagedModels.toSeq.flatten)
          .forall { case (a, b) => aten.ATen.equal(a, b) }
      ) { throw new RuntimeException("assertion failed") }
  }

}
