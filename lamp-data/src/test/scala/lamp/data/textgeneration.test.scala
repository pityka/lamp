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
import java.nio.charset.CodingErrorAction
import java.nio.charset.Charset
import scala.io.Codec
import lamp.nn.SlowTest
import lamp.nn.LSTM
import lamp.nn.SeqLinear
import lamp.nn.Seq4
import lamp.nn.Seq5
import lamp.nn.StatefulSeq5
import lamp.nn.statefulSequence

class TextGenerationSuite extends AnyFunSuite {
  val asciiSilentCharsetDecoder = Charset
    .forName("UTF8")
    .newDecoder()
    .onMalformedInput(CodingErrorAction.REPLACE)
    .onUnmappableCharacter(CodingErrorAction.REPLACE)
  test("text learning - slow - LSTM", SlowTest) {
    val trainText =
      scala.io.Source
        .fromInputStream(getClass.getResourceAsStream("/35-0.txt"))(
          Codec.apply(asciiSilentCharsetDecoder)
        )
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
        statefulSequence(
          Embedding(
            classes = vocabularSize,
            dimensions = 10,
            tOpt = tensorOptions
          ).lift,
          LSTM(
            in = 10,
            hiddenSize = 256,
            tOpt = tensorOptions
          ),
          Fun(_.relu).lift,
          SeqLinear(in = 256, out = vocabularSize, tOpt = tensorOptions).lift,
          Fun(_.logSoftMax(2)).lift
        ).unlift

      SupervisedModel(
        net,
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
        epochs = 10
      )
      .unsafeRunSync()

    assert(buffer.last < 3d)

  }
  test("text learning", SlowTest) {
    val trainText =
      scala.io.Source
        .fromInputStream(getClass.getResourceAsStream("/35-0.txt"))(
          Codec.apply(asciiSilentCharsetDecoder)
        )
        .mkString

    val (vocab, _) = Text.charsToIntegers(trainText)
    val trainTokenized = Text.charsToIntegers(trainText, vocab)
    val vocabularSize = vocab.size
    val rvocab = vocab.map(_.swap)

    val hiddenSize = 1024
    val lookAhead = 10
    val device = CPU
    val precision = SinglePrecision
    val tensorOptions = device.options(precision)
    val model = {
      val classWeights =
        ATen.ones(Array(vocabularSize), tensorOptions)
      val net =
        statefulSequence(
          Embedding(
            classes = vocabularSize,
            dimensions = 10,
            tOpt = tensorOptions
          ).lift,
          RNN(
            in = 10,
            hiddenSize = hiddenSize,
            tOpt = tensorOptions
          ),
          Fun(_.relu).lift,
          SeqLinear(in = hiddenSize, out = vocabularSize, tOpt = tensorOptions).lift,
          Fun(_.logSoftMax(2)).lift
        ).unlift

      SupervisedModel(
        net,
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
      learningRate = simple(0.0001),
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
        // checkpointFile = Some(new java.io.File("checkpoint.test")),
        // logger = Some(scribe.Logger("training"))
      )
      .unsafeRunSync()

    assert(buffer.last < 8d)

  }
  test("text generation") {
    val trainText =
      scala.io.Source
        .fromInputStream(getClass.getResourceAsStream("/35-0.txt"))(
          Codec.apply(asciiSilentCharsetDecoder)
        )
        .mkString

    val (vocab, _) = Text.charsToIntegers(trainText)
    val vocabularSize = vocab.size
    val rvocab = vocab.map(_.swap)

    val hiddenSize = 1024
    val lookAhead = 10
    val device = CPU
    val precision = SinglePrecision
    val tensorOptions = device.options(precision)

    val net =
      statefulSequence(
        Embedding(
          classes = vocabularSize,
          dimensions = 10,
          tOpt = tensorOptions
        ).lift,
        RNN(
          in = 10,
          hiddenSize = hiddenSize,
          tOpt = tensorOptions
        ),
        Fun(_.relu).lift,
        SeqLinear(in = hiddenSize, out = vocabularSize, tOpt = tensorOptions).lift,
        Fun(_.logSoftMax(2)).lift
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
        lookAhead
      )
      .use { variable =>
        IO { Text.convertIntegersToText(variable.value, rvocab) }
      }
      .unsafeRunSync()

    assert(text == Vector(" the the t"))
  }
  test("text generation - beam") {
    val trainText =
      scala.io.Source
        .fromInputStream(getClass.getResourceAsStream("/35-0.txt"))(
          Codec.apply(asciiSilentCharsetDecoder)
        )
        .mkString

    val (vocab, _) = Text.charsToIntegers(trainText)
    val vocabularSize = vocab.size
    val rvocab = vocab.map(_.swap)

    val hiddenSize = 1024
    val lookAhead = 10
    val device = CPU
    val precision = SinglePrecision
    val tensorOptions = device.options(precision)

    val net =
      statefulSequence(
        Embedding(
          classes = vocabularSize,
          dimensions = 10,
          tOpt = tensorOptions
        ).lift,
        RNN(
          in = 10,
          hiddenSize = hiddenSize,
          tOpt = tensorOptions
        ),
        Fun(_.relu).lift,
        SeqLinear(in = hiddenSize, out = vocabularSize, tOpt = tensorOptions).lift,
        Fun(_.logSoftMax(2)).lift
      )

    val channel = Resource.make(IO {
      val is = getClass.getResourceAsStream("/checkpoint.test")
      java.nio.channels.Channels.newChannel(is)
    })(v => IO { v.close })
    val trainedModel =
      Reader.loadFromChannel(net, channel, device).unsafeRunSync().right.get
    val text = Text
      .sequencePredictionBeam(
        List("time machine")
          .map(t => Text.charsToIntegers(t, vocab).map(_.toLong))
          .head,
        device,
        precision,
        trainedModel,
        lookAhead
      )
      .use { variables =>
        IO {
          variables.map(v =>
            (Text.convertIntegersToText(v._1.value, rvocab).mkString, v._2)
          )
        }
      }
      .unsafeRunSync()

    assert(
      text.map(_._1) == Seq("e the the t", "ed and and ", "ed and the ").reverse
    )
  }
}
