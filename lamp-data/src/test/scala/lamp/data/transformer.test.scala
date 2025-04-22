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
import cats.effect.kernel.Resource
import lamp.autograd.Autograd.BackpropForwardCache
import lamp.autograd.Autograd.CacheStrategy
import lamp.Device
import aten.Tensor
class TransformerSuite extends AnyFunSuite {

  def test1(id: String)(fun: Device => Unit) = {
    test(id) { fun(CPU) }
    if (aten.Tensor.hasMps) {test(id+" MPS") { fun( lamp.MPS) }}
    if (Tensor.hasCuda) {test(id + "/CUDA") { fun(CudaDevice(0)) }}
  }

  test1("clickbait") { device =>
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

    val all = Mat(
      Text
        .sentencesToPaddedMatrix(
          positives ++ negatives,
          maxLength = 15,
          pad = vocab.size,
          vocabulary = vocab
        )
        .map(_.toVec): _*
    ).T
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
            ).map(v => Resource.pure(NonEmptyBatch((v._1.constantValue,v._2))))

        val classWeights = STen.ones(List(2), device.options(precision))
        def makeModel = SupervisedModel(
          sequence(
            lamp.nn.TransformerEmbedding(
              lamp.nn
                .Embedding(
                  classes = vocab.size + 1,
                  dimensions = 45,
                  tOpt = tOpt
                ),
              positionalEmbedding = const(
                PositionalEmbedding
                  .vaswani(
                    sequenceLength = 15,
                    dimension = 45,
                    device = device,
                    precision = precision
                  )
              )
            ),
            GenericFun[Variable, (Variable, Option[STen])] { _ => _ => x =>
              (x, None)
            },
            lamp.nn.TransformerEncoder(
              numBlocks = 3,
              in = 45,
              attentionHiddenPerHeadDim = 30,
              attentionNumHeads = 3,
              mlpHiddenDim = 30,
              dropout = 0d,
              // padToken = vocab.size,
              tOpt = tOpt,
              linearized = false,
              gptOrder = false,
              causalMask = false
            ),
            GenericFun[Variable, Variable] { _ => fw => x =>
              x.view(List(x.forward(fw).shape(0), -1))
            },
            Linear(in = 675, out = 2, tOpt = tOpt),
            Fun(_=> variable => variable.logSoftMax(1))
          ),
          LossFunctions.NLL(2, classWeights)
        )
        val model = makeModel
        val model2 = makeModel
        val model3 = makeModel
        val model4 = makeModel
        
        val (_, trainedModel, _, _,_) = IOLoops
          .epochs(
            model = model,
            optimizerFactory = RAdam
              .factory(
                learningRate = simple(0.001),
                weightDecay = simple(0d)
              ),
            trainBatchesOverEpoch = makeTrainingBatch,
            validationBatchesOverEpoch = None,
            dataParallelModels = List(model2,model3,model4),
            epochs = 10,
            logger = Some(scribe.Logger("sdf"))
          )
          .unsafeRunSync()

        trainedModel.module
      }

      val accuracy = {
        implicit val fw = BackpropForwardCache.empty(CacheStrategy.AlwaysCache,Nil)
        val output =
          trainedModel.asEval.forward((testFT))
        val prediction = {
          val argm = aten.ATen.argmax(output.forward.value, 1, false)
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
