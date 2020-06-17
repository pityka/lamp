package lamp.data

import org.scalatest.funsuite.AnyFunSuite
import lamp.CPU
import lamp.SinglePrecision
import aten.ATen
import lamp.nn.Seq3
import lamp.nn.Embedding
import lamp.nn.RNN
import lamp.nn.Fun
import lamp.nn.SupervisedModel
import lamp.nn.LossFunctions
import lamp.nn.AdamW
import lamp.nn.simple
import lamp.nn.LearningRateSchedule
import cats.effect.IO
import scala.collection.mutable
import java.io.File
import cats.effect.Resource

class TextGenerationSuite extends AnyFunSuite {
  test("text learning") {
    val trainText =
      scala.io.Source
        .fromInputStream(getClass.getResourceAsStream("/35-0.txt"))
        .mkString

    val (vocab, _) = Text.charsToIntegers(trainText)
    val trainTokenized = Text.charsToIntegers(trainText, vocab)
    val vocabularSize = vocab.size
    val rvocab = vocab.map(_.swap)

    val hiddenSize = 8
    val lookAhead = 5
    val device = CPU
    val precision = SinglePrecision
    val tensorOptions = device.options(precision)
    val model = {
      val classWeights =
        ATen.ones(Array(vocabularSize), tensorOptions)
      val net =
        Seq3(
          Embedding(
            classes = vocabularSize,
            dimensions = 10,
            tOpt = tensorOptions
          ),
          RNN(
            in = 10,
            hiddenSize = hiddenSize,
            out = vocabularSize,
            dropout = 0d,
            tOpt = tensorOptions
          ),
          Fun(_.logSoftMax(2))
        )

      SupervisedModel(
        net,
        ((), None, ()),
        LossFunctions.SequenceNLL(vocabularSize, classWeights)
      )
    }

    val trainEpochs = () =>
      Text
        .minibatchesFromText(
          trainTokenized,
          64,
          lookAhead,
          device
        )

    val optimizer = AdamW.factory(
      weightDecay = simple(0.00),
      learningRate = simple(0.1),
      scheduler = LearningRateSchedule.noop,
      clip = Some(1d)
    )

    val buffer = mutable.ArrayBuffer[Double]()
    val trainedModel = IOLoops
      .epochs(
        model = model,
        optimizerFactory = optimizer,
        trainBatchesOverEpoch = trainEpochs,
        validationBatchesOverEpoch = None,
        trainingCallback = new TrainingCallback {

          override def apply(trainingLoss: Double, batchCount: Int): Unit = {
            buffer.append(trainingLoss)
          }

        },
        epochs = 1
      )
      .unsafeRunSync()

    assert(buffer.last < 8d)

  }
  test("text generation") {
    val trainText =
      scala.io.Source
        .fromInputStream(getClass.getResourceAsStream("/35-0.txt"))
        .mkString

    val (vocab, _) = Text.charsToIntegers(trainText)
    val vocabularSize = vocab.size
    val rvocab = vocab.map(_.swap)

    val hiddenSize = 64
    val lookAhead = 5
    val device = CPU
    val precision = SinglePrecision
    val tensorOptions = device.options(precision)

    val net =
      Seq3(
        Embedding(
          classes = vocabularSize,
          dimensions = 10,
          tOpt = tensorOptions
        ),
        RNN(
          in = 10,
          hiddenSize = hiddenSize,
          out = vocabularSize,
          dropout = 0d,
          tOpt = tensorOptions
        ),
        Fun(_.logSoftMax(2))
      )

    val channel = Resource.make(IO {
      val is = getClass.getResourceAsStream("/checkpoint.test")
      java.nio.channels.Channels.newChannel(is)
    })(v => IO { v.close })
    val trainedModel =
      Reader.loadFromChannel(net, channel, device).unsafeRunSync().right.get
    val text = Text
      .sequencePrediction(
        List("time machine").map(t =>
          Text.charsToIntegers(t, vocab).map(_.toLong)
        ),
        device,
        precision,
        trainedModel,
        (
          (),
          None,
          ()
        ),
        lookAhead
      )
      .use { variable =>
        IO { Text.convertIntegersToText(variable.value, rvocab) }
      }
      .unsafeRunSync()

    assert(text == Vector(" t al"))
  }
}
