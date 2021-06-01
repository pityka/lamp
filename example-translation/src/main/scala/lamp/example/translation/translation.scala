package lamp.example.translation

import lamp.CudaDevice
import lamp.CPU
import lamp.nn.SupervisedModel
import lamp.nn.LossFunctions
import java.io.File
import lamp.nn.AdamW
import lamp.nn.simple
import cats.effect.Resource
import cats.effect.IO
import lamp.nn.Fun
import lamp.DoublePrecision
import lamp.SinglePrecision
import java.nio.charset.CodingErrorAction
import scala.io.Codec
import lamp.nn.Embedding
import lamp.nn.SeqLinear
import lamp.nn.LSTM
import lamp.nn.statefulSequence
import java.nio.charset.StandardCharsets
import lamp.nn.Seq2SeqWithAttention
import lamp.Scope
import lamp.data.Text
import lamp.data.Reader
import lamp.data.IOLoops
import lamp.STen
import cats.effect.unsafe.implicits.global

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
      endOfSentence: Char,
      startOfSentence: Char
  ) = {
    val spl = line.split("\\t+")
    assert(spl.size == 2)
    val source = spl(0)
    val target = spl(1)
    val source1 =
      (startOfSentence.toString + source + endOfSentence)
    val target1 =
      (startOfSentence.toString + target + endOfSentence)
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
      opt[Unit]("gpu").action((_, c) => c.copy(cuda = true)),
      opt[Unit]("single").action((_, c) => c.copy(singlePrecision = true)),
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
      Scope.root { implicit scope =>
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

        val (vocab1, _) =
          Text.charsToIntegers(trainText + testText.getOrElse(""))
        val vocab =
          vocab1 ++ List(
            '#' -> vocab1.size,
            '|' -> (vocab1.size + 1),
            '*' -> (vocab1.size + 2),
            '=' -> (vocab1.size + 3)
          )
        val trainTokenized = scala.io.Source
          .fromString(trainText)
          .getLines()
          .toVector
          .map { line =>
            val (feature, target) = Translation.prepare(
              line,
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
            .getLines()
            .toVector
            .map { line =>
              val (feature, target) = Translation.prepare(
                line,
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

        val hiddenSize = 256
        val lookAhead = 70
        val embeddingDimension = 8
        val device = if (config.cuda) CudaDevice(0) else CPU
        val precision =
          if (config.singlePrecision) SinglePrecision else DoublePrecision
        val tensorOptions = device.options(precision)
        val classWeights =
          STen.ones(List(vocabularSize), tensorOptions)
        val encoder = statefulSequence(
          Embedding(
            classes = vocabularSize,
            dimensions = embeddingDimension,
            tOpt = tensorOptions
          ).lift,
          LSTM(
            in = embeddingDimension,
            hiddenSize = hiddenSize,
            tOpt = tensorOptions
          )
        )

        val model = {

          val decoder = statefulSequence(
            LSTM(
              in = embeddingDimension + hiddenSize,
              hiddenSize = hiddenSize,
              tOpt = tensorOptions
            ),
            Fun(implicit scope => _.relu).lift,
            SeqLinear
              .apply(
                in = hiddenSize,
                out = vocabularSize,
                tOpt = tensorOptions
              )
              .lift,
            Fun(implicit scope => _.logSoftMax(2)).lift
          )

          val destinationEmbedding = Embedding(
            classes = vocabularSize,
            dimensions = embeddingDimension,
            tOpt = tensorOptions
          )
          val net1 =
            Seq2SeqWithAttention(
              destinationEmbedding,
              encoder.mapState { case (_, lstmState) =>
                (lstmState, (), (), ())
              },
              decoder,
              vocab('#').toLong
            )(_._1.get._1).unlift

          config.checkpointLoad
            .foreach { load =>
              scribe.info(s"Loading parameters from file $load")
              Reader
                .loadFromFile(net1, new File(load), device, false)
                .unsafeRunSync()
            }

          scribe.info("Learnable parameters: " + net1.learnableParameters)
          SupervisedModel(
            net1,
            LossFunctions
              .SequenceNLL(vocabularSize, classWeights, ignore = vocab('#'))
          )
        }
        val rng = org.saddle.spire.random.rng.Cmwc5.apply()
        val trainEpochs = () =>
          Text
            .minibatchesForSeq2Seq(
              trainTokenized,
              config.trainBatchSize,
              lookAhead,
              vocab('#'),
              rng
            )
        val testEpochs = testTokenized.map { t => () =>
          Text
            .minibatchesForSeq2Seq(
              t,
              config.validationBatchSize,
              lookAhead,
              vocab('#'),
              rng
            )
        }

        val optimizer = AdamW.factory(
          weightDecay = simple(0.00),
          learningRate = simple(config.learningRate),
          clip = Some(1d)
        )

        val (_, trainedModel, _) = IOLoops
          .epochs(
            model = model,
            optimizerFactory = optimizer,
            trainBatchesOverEpoch = trainEpochs,
            validationBatchesOverEpoch = testEpochs,
            epochs = config.epochs,
            logger = Some(scribe.Logger("training")),
            validationFrequency = 1
          )
          .unsafeRunSync()
        scribe.info("Training done.")

        config.query.foreach { prefix =>
          val text = IO(Scope.free)
            .bracket { implicit scope =>
              IO {
                val warmupBatch = Text
                  .makePredictionBatch(
                    List(prefix)
                      .map(t => Text.charsToIntegers(t, vocab).map(_.toLong)),
                    device
                  )

                val enc = trainedModel.module.statefulModule.encoder

                val (encOut, encState) =
                  enc.forward(warmupBatch -> enc.initState)

                val dec =
                  trainedModel.module.statefulModule
                    .attentionDecoder(encOut, warmupBatch)
                    .withInit(encState)

                val beams = Text
                  .sequencePredictionBeam(
                    Text.charsToIntegers("=", vocab).map(_.toLong),
                    device,
                    dec,
                    lookAhead,
                    vocab('='),
                    vocab('*')
                  )
                beams.map(_._1).map { v =>
                  Text
                    .convertIntegersToText(
                      v,
                      rvocab
                    )
                    .mkString
                }
              }
            }(b => IO(b.release()))
            .unsafeRunSync()

          scribe.info(
            s"Answer to query text follows (from prefix '$prefix'): \n\n" + text
          )
        }
      }
    case _ =>
    // arguments are bad, error message will have been displayed
  }
  scribe.info("END")
}
