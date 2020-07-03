package lamp.example.translation

import lamp.CudaDevice
import lamp.CPU
import lamp.nn.SupervisedModel
import aten.ATen
import lamp.nn.Module
import lamp.nn.LossFunctions
import lamp.data.{Reader, Text, IOLoops, Peek}
import java.io.File
import lamp.data.BatchStream
import lamp.nn.AdamW
import lamp.nn.LearningRateSchedule
import lamp.nn.simple
import lamp.nn.RNN
import cats.effect.Resource
import cats.effect.IO
import lamp.data.TrainingCallback
import lamp.data.ValidationCallback
import lamp.nn.Sequential
import lamp.nn.Fun
import aten.Tensor
import lamp.util.NDArray
import lamp.syntax
import lamp.nn.Seq2
import lamp.nn.StatefulModule
import lamp.autograd.Variable
import lamp.autograd.const
import lamp.nn.Seq3
import lamp.nn.Seq4
import lamp.DoublePrecision
import lamp.FloatingPointPrecision
import lamp.SinglePrecision
import lamp.nn.GRU
import java.nio.charset.CodingErrorAction
import java.nio.charset.Charset
import scala.io.Codec
import lamp.nn.Embedding
import lamp.nn.SeqLinear
import lamp.nn.Seq5
import lamp.nn.LSTM
import lamp.nn.statefulSequence
import lamp.nn.Seq2Seq
import lamp.nn.GenericModule
import java.nio.charset.StandardCharsets
import lamp.nn.Seq2SeqWithAttention
import lamp.nn.Linear

case class CliConfig(
    trainData: String = "",
    testData: Option[String] = None,
    cuda: Boolean = false,
    singlePrecision: Boolean = false,
    trainBatchSize: Int = 256,
    validationBatchSize: Int = 256,
    epochs: Int = 1000,
    learningRate: Double = 0.0001,
    dropout: Double = 0.0,
    checkpointSave: Option[String] = None,
    checkpointLoad: Option[String] = None,
    query: Option[String] = None
)

object Translation {
  def prepare(
      line: String,
      pad: Char,
      endOfSentence: Char,
      startOfSentence: Char
  ) = {
    val spl = line.split("\\t+")
    assert(spl.size == 2)
    val source = spl(0)
    val target = spl(1)
    val source1 =
      (startOfSentence + source + endOfSentence)
    val target1 =
      (startOfSentence + target + endOfSentence)
    (source1, target1)
  }
}

object Train extends App {
  scribe.info("Logger start")
  import scopt.OParser
  val builder = OParser.builder[CliConfig]
  val parser1 = {
    import builder._
    OParser.sequence(
      opt[String]("train-data")
        .action((x, c) => c.copy(trainData = x))
        .text("path to train data ")
        .required(),
      opt[String]("test-data")
        .action((x, c) => c.copy(testData = Some(x)))
        .text("path to validation data"),
      opt[Unit]("gpu").action((x, c) => c.copy(cuda = true)),
      opt[Unit]("single").action((x, c) => c.copy(singlePrecision = true)),
      opt[Int]("train-batch").action((x, c) => c.copy(trainBatchSize = x)),
      opt[Int]("validation-batch").action((x, c) =>
        c.copy(validationBatchSize = x)
      ),
      opt[Int]("epochs").action((x, c) => c.copy(epochs = x)),
      opt[Double]("learning-rate").action((x, c) => c.copy(learningRate = x)),
      opt[Double]("dropout").action((x, c) => c.copy(dropout = x)),
      opt[String]("checkpoint-save").action((x, c) =>
        c.copy(checkpointSave = Some(x))
      ),
      opt[String]("checkpoint-load").action((x, c) =>
        c.copy(checkpointLoad = Some(x))
      ),
      opt[String]("query").action((x, c) => c.copy(query = Some(x)))
    )

  }

  OParser.parse(parser1, args, CliConfig()) match {
    case Some(config) =>
      scribe.info(s"Config: $config")

      val charsetDecoder = StandardCharsets.UTF_8
        .newDecoder()
        .onMalformedInput(CodingErrorAction.REPLACE)
        .onUnmappableCharacter(CodingErrorAction.REPLACE)

      val trainText = Resource
        .make(IO {
          scala.io.Source.fromFile(new File(config.trainData))(
            Codec.apply(charsetDecoder)
          )
        })(s => IO { s.close })
        .use(s => IO(s.mkString))
        .unsafeRunSync()

      val testText = config.testData.map { t =>
        Resource
          .make(IO {
            scala.io.Source.fromFile(new File(t))(
              Codec.apply(charsetDecoder)
            )
          })(s => IO { s.close })
          .use(s => IO(s.mkString))
          .unsafeRunSync()
      }

      val (vocab1, _) = Text.charsToIntegers(trainText + testText.getOrElse(""))
      val vocab =
        vocab1 + ('#' -> vocab1.size, '|' -> (vocab1.size + 1), '*' -> (vocab1.size + 2), '=' -> (vocab1.size + 3))
      val trainTokenized = scala.io.Source
        .fromString(trainText)
        .getLines
        .toVector
        .map { line =>
          val (feature, target) = Translation.prepare(
            line,
            pad = '#',
            endOfSentence = '*',
            startOfSentence = '='
          )
          (
            Text
              .charsToIntegers(
                feature,
                vocab
              )
              .map(_.toLong),
            Text
              .charsToIntegers(
                target,
                vocab
              )
              .map(_.toLong)
          )
        }
      val testTokenized = testText.map { t =>
        scala.io.Source
          .fromString(t)
          .getLines
          .toVector
          .map { line =>
            val (feature, target) = Translation.prepare(
              line,
              pad = '#',
              endOfSentence = '*',
              startOfSentence = '='
            )
            (
              Text
                .charsToIntegers(
                  feature,
                  vocab
                )
                .map(_.toLong),
              Text
                .charsToIntegers(
                  target,
                  vocab
                )
                .map(_.toLong)
            )
          }
      }
      val vocabularSize = vocab.size
      val rvocab = vocab.map(_.swap)
      scribe.info(
        s"Vocabulary size $vocabularSize, tokenized length of train ${trainTokenized.size}, test ${testTokenized.size}"
      )

      val hiddenSize = 1024
      val lookAhead = 100
      val device = if (config.cuda) CudaDevice(0) else CPU
      val precision =
        if (config.singlePrecision) SinglePrecision else DoublePrecision
      val tensorOptions = device.options(precision)
      val model = {
        val classWeights =
          ATen.ones(Array(vocabularSize), tensorOptions)

        val encoder = statefulSequence(
          Embedding(
            classes = vocabularSize,
            dimensions = 20,
            tOpt = tensorOptions
          ).lift,
          LSTM(
            in = 20,
            hiddenSize = hiddenSize,
            tOpt = tensorOptions
          )
        )
        val contextModule =
          Linear(in = hiddenSize, out = 20, tOpt = tensorOptions)
        val decoder = statefulSequence(
          LSTM(
            in = 20 + 20,
            hiddenSize = hiddenSize,
            tOpt = tensorOptions
          ),
          Fun(_.relu).lift,
          SeqLinear
            .apply(
              in = hiddenSize,
              out = vocabularSize,
              tOpt = tensorOptions
            )
            .lift,
          Fun(_.logSoftMax(2)).lift
        )

        val destinationEmbedding = Embedding(
          classes = vocabularSize,
          dimensions = 20,
          tOpt = tensorOptions
        )
        val net1 =
          Seq2SeqWithAttention(
            destinationEmbedding,
            encoder.mapState {
              case (_, lstmState) => (lstmState, (), (), ())
            },
            decoder,
            contextModule
          )(_._1.get._1).unlift

        val net =
          config.checkpointLoad
            .map { load =>
              scribe.info(s"Loading parameters from file $load")
              Reader
                .loadFromFile(net1, new File(load), device)
                .unsafeRunSync()
                .right
                .get
            }
            .getOrElse(net1)

        scribe.info("Learnable parameters: " + net.learnableParameters)
        SupervisedModel(
          net,
          LossFunctions
            .SequenceNLL(vocabularSize, classWeights, ignore = vocab('#'))
        )
      }

      val trainEpochs = () =>
        Text
          .minibatchesForSeq2Seq(
            trainTokenized,
            config.trainBatchSize,
            lookAhead,
            vocab('#'),
            device
          )
      val testEpochs = testTokenized.map { t => () =>
        Text
          .minibatchesForSeq2Seq(
            t,
            config.validationBatchSize,
            lookAhead,
            vocab('#'),
            device
          )
      }

      val optimizer = AdamW.factory(
        weightDecay = simple(0.00),
        learningRate = simple(config.learningRate),
        scheduler = LearningRateSchedule.cyclicSchedule(10d, 200L),
        clip = Some(1d)
      )

      val validationCallback = new ValidationCallback {

        override def apply(
            validationOutput: Tensor,
            validationTarget: Tensor,
            validationLoss: Double,
            epochCount: Long
        ): Unit = {
          if (true) {
            val targetString =
              Text.convertIntegersToText(validationTarget, rvocab)
            val outputString =
              Text.convertLogitsToText(validationOutput, rvocab)
            scribe.info(
              (targetString zip outputString)
                .map(x => "'" + x._1 + "'  -->  '" + x._2 + "'")
                .mkString("\n")
            )
          }

        }

      }

      val trainedModel = IOLoops
        .epochs(
          model = model,
          optimizerFactory = optimizer,
          trainBatchesOverEpoch = trainEpochs,
          validationBatchesOverEpoch = testEpochs,
          epochs = config.epochs,
          trainingCallback = TrainingCallback.noop,
          validationCallback = validationCallback,
          checkpointFile = config.checkpointSave.map(s => new File(s)),
          minimumCheckpointFile =
            config.checkpointSave.map(s => new File(s + ".min")),
          logger = Some(scribe.Logger("training")),
          logFrequency = 10,
          validationFrequency = 1
        )
        .unsafeRunSync()
      scribe.info("Training done.")

      config.query.foreach { prefix =>
        val text = Text
          .makePredictionBatch(
            List(prefix).map(t => Text.charsToIntegers(t, vocab).map(_.toLong)),
            device,
            precision
          )
          .use { warmupBatch =>
            val enc = trainedModel.module.statefulModule.encoder

            val (encOut, encState) =
              enc.forward(const(warmupBatch) -> enc.initState)

            val dec =
              trainedModel.module.statefulModule.decoder.withInit(encState)

            Text
              .sequencePredictionBeam(
                Text.charsToIntegers("=", vocab).map(_.toLong),
                device,
                precision,
                dec,
                lookAhead,
                vocab('='),
                vocab('*')
              )
              .use { variable =>
                IO {
                  variable.map(_._1).map { v =>
                    Text
                      .convertIntegersToText(
                        v.value,
                        rvocab
                      )
                      .mkString
                  }
                }
              }
          }
          .unsafeRunSync()

        scribe.info(
          s"Answer to query text follows (from prefix '$prefix'): \n\n" + text
        )
      }

    case _ =>
    // arguments are bad, error message will have been displayed
  }
  scribe.info("END")
}
