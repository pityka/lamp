package lamp.example.timemachine

import lamp.CudaDevice
import lamp.CPU
import lamp.nn.SupervisedModel
import lamp.nn.LossFunctions
import lamp.data.{Reader, Text, IOLoops}
import java.io.File
import lamp.nn.AdamW
import lamp.nn.simple
import cats.effect.Resource
import cats.effect.IO
import lamp.data.TrainingCallback
import lamp.data.ValidationCallback
import lamp.nn.Fun
import lamp.DoublePrecision
import lamp.SinglePrecision
import java.nio.charset.CodingErrorAction
import java.nio.charset.Charset
import scala.io.Codec
import lamp.nn.Embedding
import lamp.nn.SeqLinear
import lamp.nn.LSTM
import lamp.nn.statefulSequence
import lamp.Scope
import lamp.STen

case class CliConfig(
    trainData: String = "",
    testData: String = "",
    cuda: Boolean = false,
    singlePrecision: Boolean = false,
    trainBatchSize: Int = 256,
    validationBatchSize: Int = 256,
    epochs: Int = 1000,
    learningRate: Double = 0.0001,
    dropout: Double = 0.0,
    checkpointSave: Option[String] = None,
    checkpointLoad: Option[String] = None,
    predictionPrefix: Option[String] = None
)

object Train extends App {
  scribe.info("Logger start")
  import scopt.OParser
  val builder = OParser.builder[CliConfig]
  val parser1 = {
    import builder._
    OParser.sequence(
      opt[String]("train-data")
        .action((x, c) => c.copy(trainData = x))
        .text("path to cifar100 binary train data")
        .required(),
      opt[String]("test-data")
        .action((x, c) => c.copy(testData = x))
        .text("path to cifar100 binary test data")
        .required(),
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
      opt[String]("prefix").action((x, c) => c.copy(predictionPrefix = Some(x)))
    )

  }

  OParser.parse(parser1, args, CliConfig()) match {
    case Some(config) =>
      scribe.info(s"Config: $config")
      Scope.root { implicit scope =>
        val asciiSilentCharsetDecoder = Charset
          .forName("UTF8")
          .newDecoder()
          .onMalformedInput(CodingErrorAction.REPLACE)
          .onUnmappableCharacter(CodingErrorAction.REPLACE)

        val trainText = Resource
          .make(IO {
            scala.io.Source.fromFile(new File(config.trainData))(
              Codec.apply(asciiSilentCharsetDecoder)
            )
          })(s => IO { s.close })
          .use(s => IO(s.mkString))
          .unsafeRunSync()

        val testText = Resource
          .make(IO {
            scala.io.Source.fromFile(new File(config.testData))
          })(s => IO { s.close })
          .use(s => IO(s.mkString))
          .unsafeRunSync()

        val (vocab, _) = Text.charsToIntegers(trainText + testText)
        val trainTokenized = Text.charsToIntegers(trainText, vocab)
        val testTokenized = Text.charsToIntegers(testText, vocab)
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
            STen.ones(Array(vocabularSize), tensorOptions)
          val net1 =
            statefulSequence(
              Embedding(
                classes = vocabularSize,
                dimensions = 20,
                tOpt = tensorOptions
              ).lift,
              LSTM(
                in = 20,
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
            ).unlift

          config.checkpointLoad
            .foreach { load =>
              scribe.info(s"Loading parameters from file $load")
              Reader
                .loadFromFile(net1, new File(load), device)
                .unsafeRunSync()
                .right
                .get
            }

          scribe.info("Learnable parameters: " + net1.learnableParameters)
          SupervisedModel(
            net1,
            LossFunctions.SequenceNLL(vocabularSize, classWeights)
          )
        }
        val rng = org.saddle.spire.random.rng.Cmwc5.apply
        val trainEpochs = () =>
          Text
            .minibatchesFromText(
              trainTokenized,
              config.trainBatchSize,
              lookAhead,
              device,
              rng
            )
        val testEpochs = () =>
          Text
            .minibatchesFromText(
              testTokenized,
              config.validationBatchSize,
              lookAhead,
              device,
              rng
            )

        val optimizer = AdamW.factory(
          weightDecay = simple(0.00),
          learningRate = simple(config.learningRate),
          clip = Some(1d)
        )

        val validationCallback = new ValidationCallback {

          override def apply(
              validationOutput: STen,
              validationTarget: STen,
              validationLoss: Double,
              epochCount: Long
          ): Unit = {
            if (false) {
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

        val (_, trainedModel) = IOLoops
          .epochs(
            model = model,
            optimizerFactory = optimizer,
            trainBatchesOverEpoch = trainEpochs,
            validationBatchesOverEpoch = Some(testEpochs),
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

        config.predictionPrefix.foreach { prefix =>
          val text1 = Text
            .sequencePrediction(
              List(prefix).map(t =>
                Text.charsToIntegers(t, vocab).map(_.toLong)
              ),
              device,
              trainedModel.module.statefulModule,
              lookAhead
            )
          val text = Text.convertIntegersToText(text1, rvocab)

          scribe.info(
            s"Hallucinated text follows (from prefix '$prefix'): \n\n" + prefix + text
              .mkString("\n")
          )
        }
      }
    case _ =>
    // arguments are bad, error message will have been displayed
  }
  scribe.info("END")
}
