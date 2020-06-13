package lamp.example.timemachine

import lamp.data.CudaDevice
import lamp.data.CPU
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

case class CliConfig(
    trainData: String = "",
    testData: String = "",
    cuda: Boolean = false,
    batchSize: Int = 256,
    epochs: Int = 1000,
    learningRate: Double = 0.001,
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
      opt[Int]("batch").action((x, c) => c.copy(batchSize = x)),
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
          scala.io.Source.fromFile(new File(config.trainData))
        })(s => IO { s.close })
        .use(s => IO(s.mkString))
        .unsafeRunSync()

      val (vocab, trainTokenized) = Text.englishToIntegers(trainText)
      val testTokenized = Text.englishToIntegers(testText, vocab)
      val vocabularSize = vocab.size + 1
      scribe.info(
        s"Vocabulary size $vocabularSize, tokenized length of train ${trainTokenized.size}, test ${testTokenized.size}"
      )

      val device = if (config.cuda) CudaDevice(0) else CPU
      val model: SupervisedModel[(Option[Variable], Unit)] = {
        val classWeights = ATen.ones(Array(vocabularSize), device.options)
        val net: StatefulModule[(Option[Variable], Unit)] =
          Seq2(
            RNN(
              in = vocabularSize,
              hiddenSize = 64,
              out = vocabularSize,
              dropout = config.dropout,
              tOpt = device.options
            ),
            Fun(_.logSoftMax(2))
          )

        scribe.info("Learnable parametes: " + net.learnableParameters)
        scribe.info("parameters: " + net.parameters.mkString("\n"))
        SupervisedModel(
          net,
          (None, ()),
          LossFunctions.SequenceNLL(vocabularSize, classWeights)
        )
      }

      val lookAhead = 10

      val trainEpochs = () =>
        Text
          .minibatchesFromText(
            trainTokenized,
            config.batchSize,
            lookAhead,
            device
          )
          .map(BatchStream.oneHotFeatures(vocabularSize))
      val testEpochs = () =>
        Text
          .minibatchesFromText(
            testTokenized,
            config.batchSize,
            lookAhead,
            device
          )
          .map(BatchStream.oneHotFeatures(vocabularSize))

      val optimizer = AdamW.factory(
        weightDecay = simple(0.00),
        learningRate = simple(config.learningRate),
        scheduler = LearningRateSchedule.cyclicSchedule(10d, 200L),
        clip = Some(1d)
      )

      val trainedModel = IOLoops
        .epochs(
          model = model,
          optimizerFactory = optimizer,
          trainBatchesOverEpoch = trainEpochs,
          validationBatchesOverEpoch = testEpochs,
          epochs = config.epochs,
          trainingCallback = TrainingCallback.noop,
          validationCallback = ValidationCallback.noop,
          checkpointFile = None, //config.checkpointSave.map(s => new File(s)),
          minimumCheckpointFile = None,
          checkpointFrequency = 10,
          logger = Some(scribe.Logger("training"))
        )
        .unsafeRunSync()

      val exampleText = "time"
      val tokenized = Text.englishToIntegers(exampleText, vocab).map(_.toLong)
      val predicted = Text.sequencePrediction(
        List(tokenized),
        device,
        trainedModel.module,
        (Some(const(ATen.zeros(Array(1, 64), device.options))), ()),
        vocabularSize,
        lookAhead
      )
      val rvocab = vocab.map(_.swap)
      val text = predicted
        .use { variable => IO { Text.convertToText(variable.value, rvocab) } }
        .unsafeRunSync()
      scribe.info(text.toString)

    case _ =>
    // arguments are bad, error message will have been displayed
  }
  scribe.info("END")
}
