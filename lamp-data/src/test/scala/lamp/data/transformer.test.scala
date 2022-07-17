package lamp.data

import org.scalatest.funsuite.AnyFunSuite
import org.saddle._
import lamp.nn._
import lamp.{CPU, CudaDevice}
import lamp.SinglePrecision
import lamp.Scope
import lamp.autograd.Variable
import lamp.autograd.const
import lamp.STen
import scala.io.Codec
import cats.effect.unsafe.implicits.global

class TransformerSuite extends AnyFunSuite {
  def test1(id: String)(fun: Boolean => Unit) = {
    test(id) { fun(false) }
    test(id + "/CUDA", CudaTest) { fun(true) }
  }

  test1("clickbait") { cuda =>
    val device = if (cuda) CudaDevice(0) else CPU
    val precision = SinglePrecision

    val positives = scala.io.Source
      .fromInputStream(getClass.getResourceAsStream("/clickbait_data"))(
        Codec.UTF8
      )
      .getLines()
      .toVector
    val negatives = scala.io.Source
      .fromInputStream(getClass.getResourceAsStream("/non_clickbait_data"))(
        Codec.UTF8
      )
      .getLines()
      .toVector

    val (vocab, _) = Text.charsToIntegers((positives ++ negatives).mkString)

    val all = Mat(Text.sentencesToPaddedMatrix(
      positives ++ negatives,
      maxLength = 15,
      pad = vocab.size,
      vocabulary = vocab
    ).map(_.toVec):_*).T
    val target = vec.ones(positives.size) concat vec.zeros(negatives.size)
    val shuffle =
      scala.util.Random.shuffle(0 until all.numRows toVector).toArray
    val shuffledF =
      all.row(shuffle)
    val shuffledT = target.take(shuffle)
    val testIdx = array.range(0, 2000)
    val testTarget = shuffledT.take(testIdx)
    val testF = shuffledF.row(testIdx)
    val trainIdx = array.range(2000, 30000)
    val trainTarget = shuffledT.take(trainIdx)
    val trainF = shuffledF.row(trainIdx)

    val rng = new scala.util.Random

    Scope.root { implicit scope =>
      val tOpt = device.options(precision)
      val trainFT = lamp.saddle.fromLongMat(trainF.map(_.toLong), CPU)
      val trainTargetT = lamp.saddle.fromLongVec(trainTarget.map(_.toLong), CPU)
      val testFT = lamp.saddle.fromLongMat(testF.map(_.toLong), device)

      val trainedModel = Scope { implicit scope =>
        val makeTrainingBatch =
          (_: IOLoops.TrainingLoopContext) =>
            BatchStream.minibatchesFromFull(
              1024,
              false,
              trainFT,
              trainTargetT,
              rng
            )

        val classWeights = STen.ones(List(2), device.options(precision))
        val model = SupervisedModel(
          sequence(
            lamp.nn.TransformerEmbedding(
              lamp.nn
                .Embedding(
                  classes = vocab.size + 1,
                  dimensions = 30,
                  tOpt = tOpt
                ),
              addPositionalEmbedding = false,
              positionalEmbedding = const(
                PositionalEmbedding
                  .simpleSequence(15, 30, 15, device, precision)
              )
            ),
            lamp.nn.TransformerEncoder(
              numBlocks = 3,
              in = 45,
              attentionHiddenPerHeadDim = 30,
              attentionNumHeads = 3,
              mlpHiddenDim = 30,
              dropout = 0d,
              padToken = vocab.size,
              tOpt = tOpt,
              linearized = false
            ),
            GenericFun[Variable, Variable] { implicit scope => x =>
              x.view(List(x.shape(0), -1))
            },
            Linear(in = 675, out = 2, tOpt = tOpt),
            Fun(implicit scope => variable => variable.logSoftMax(1))
          ),
          LossFunctions.NLL(2, classWeights)
        )

        val (_, trainedModel, _, _) = IOLoops
          .withSWA(
            model = model,
            optimizerFactory = RAdam
              .factory(
                learningRate = simple(0.01),
                weightDecay = simple(0d)
              ),
            trainBatchesOverEpoch = makeTrainingBatch,
            warmupEpochs = 5,
            swaEpochs = 5,
            logger = Some(scribe.Logger("sdf"))
          )
          .unsafeRunSync()

        trainedModel.module
      }

      val accuracy = {
        val output =
          trainedModel.asEval.forward(lamp.autograd.const(testFT))
        val prediction = {
          val argm = aten.ATen.argmax(output.value.value, 1, false)
          val r = lamp.saddle.SaddleTensorHelpers.toLongMat(argm).toVec
          argm.release
          r
        }
        val correct =
          prediction.zipMap(testTarget)((a, b) => if (a == b) 1d else 0d)
        correct.mean2
      }
      println(accuracy)
      assert(accuracy > 0.6)
      ()
    }

  }

}
