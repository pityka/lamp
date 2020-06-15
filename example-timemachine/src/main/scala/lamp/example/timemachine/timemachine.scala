package lamp.example.timemachine

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
    checkpointLoad: Option[String] = None
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
      )
    )

  }

  OParser.parse(parser1, args, CliConfig()) match {
    case Some(config) =>
      scribe.info(s"Config: $config")

      val trainText = Resource
        .make(IO {
          scala.io.Source.fromFile(new File(config.trainData))
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

      val hiddenSize = 64
      val lookAhead = 5
      val device = if (config.cuda) CudaDevice(0) else CPU
      val precision =
        if (config.singlePrecision) SinglePrecision else DoublePrecision
      val tensorOptions = device.options(precision)
      val model = {
        val classWeights =
          ATen.ones(Array(vocabularSize), tensorOptions)
        val net1 =
          Seq2(
            GRU(
              in = vocabularSize,
              hiddenSize = hiddenSize,
              out = vocabularSize,
              dropout = config.dropout,
              tOpt = tensorOptions
            ),
            Fun(_.logSoftMax(2))
          )

        val net = config.checkpointLoad
          .map { load => Reader.loadFromFile(net1, new File(load)) }
          .getOrElse(net1)

        scribe.info("Learnable parametes: " + net.learnableParameters)
        scribe.info("parameters: " + net.parameters.mkString("\n"))
        SupervisedModel(
          net,
          (None, ()),
          LossFunctions.SequenceNLL(vocabularSize, classWeights)
        )
      }

      val trainEpochs = () =>
        Text
          .minibatchesFromText(
            trainTokenized,
            config.trainBatchSize,
            lookAhead,
            device
          )
          .map(BatchStream.oneHotFeatures(vocabularSize, precision))
      val testEpochs = () =>
        Text
          .minibatchesFromText(
            testTokenized,
            config.validationBatchSize,
            lookAhead,
            device
          )
          .map(BatchStream.oneHotFeatures(vocabularSize, precision))

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

      val exampleTexts =
        List("time", "good", "mach", "morn", "best", "then", "cand")
      val tokenized =
        exampleTexts.map(t => Text.charsToIntegers(t, vocab).map(_.toLong))
      val predicted = Text.sequencePrediction(
        tokenized,
        device,
        precision,
        trainedModel.module,
        (
          None,
          ()
        ),
        vocabularSize,
        lookAhead
      )

      val text = predicted
        .use { variable =>
          IO { Text.convertLogitsToText(variable.value, rvocab) }
        }
        .unsafeRunSync()
      scribe.info(text.toString)

    case _ =>
    // arguments are bad, error message will have been displayed
  }
  scribe.info("END")
}
