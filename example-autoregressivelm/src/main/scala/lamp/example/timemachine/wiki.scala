package lamp.example.autoregressivelm

import lamp._
import lamp.nn._
import lamp.data._

import java.nio.charset.CodingErrorAction
import java.nio.charset.Charset
import cats.effect.unsafe.implicits.global
import java.io.FileInputStream
import java.util.zip.ZipInputStream
import java.io.File
import cats.effect.IO

case class CliConfig(
    gpus: Seq[Int] = Nil,
    wiki2: String = "",
    trainBatchSize: Int = 32,
    validationBatchSize: Int = 32,
    epochs: Int = 1000,
    learningRate: Double = 0.0001,
    dropout: Double = 0.0,
    numBatchesPerEpoch: Int = 100,
    maxLength: Int = 128,
    checkpointSave: Option[String] = None,
    checkpointLoad: Option[String] = None,
    extend: Option[String] = None
)

object Train extends App {
  scribe.info("Logger start")

  def allocateModel(device: Device, maxLength: Int)(implicit scope: Scope) = {
    val tensorOptions = device.options(SinglePrecision)
    val net = lamp.nn.languagemodel.LanguageModelLoss.apply(
      maxLength = maxLength,
      vocabularySize = 256,
      numBlocks = 12,
      embeddingDim = 768,
      attentionHiddenPerHeadDim = 64,
      attentionNumHeads = 12,
      encoderMlpHiddenDim = 768,
      dropout = 0d,
      padToken = 255L,
      tOpt = tensorOptions,
      linearized = false
    )
    SupervisedModel(net, LossFunctions.Identity)
  }

  import scopt.OParser
  val builder = OParser.builder[CliConfig]
  val parser1 = {
    import builder._
    OParser.sequence(
      opt[Seq[Int]]("gpus").action((x, c) => c.copy(gpus = x)),
      opt[String]("wiki2").action((x, c) => c.copy(wiki2 = x)),
      opt[Int]("train-batch").action((x, c) => c.copy(trainBatchSize = x)),
      opt[Int]("validation-batch").action((x, c) =>
        c.copy(validationBatchSize = x)
      ),
      opt[String]("extend").action((x, c) => c.copy(extend = Some(x))),
      opt[Int]("epochs").action((x, c) => c.copy(epochs = x)),
      opt[Int]("batches-per-epoch").action((x, c) =>
        c.copy(numBatchesPerEpoch = x)
      ),
      opt[Int]("max-length").action((x, c) => c.copy(maxLength = x)),
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

  val asciiSilentCharsetDecoder = Charset
    .forName("UTF8")
    .newDecoder()
    .onMalformedInput(CodingErrorAction.REPLACE)
    .onUnmappableCharacter(CodingErrorAction.REPLACE)

  def readFromZip(zip: String): Map[String, Array[Byte]] = {
    val zis = new ZipInputStream(new FileInputStream(zip));
    var ze = zis.getNextEntry();
    val map = scala.collection.mutable.Map.empty[String, Array[Byte]]
    while (ze != null) {
      val name = ze.getName()
      val str = scala.io.Source
        .fromInputStream(zis)(asciiSilentCharsetDecoder)
        .getLines()
        .toSeq
        .mkString("\n")
        .getBytes("US-ASCII")
      map += ((name, str))
      ze = zis.getNextEntry

    }
    map.toMap

  }

  OParser.parse(parser1, args, CliConfig()) match {
    case Some(config) if config.extend.isEmpty =>
      scribe.info(s"Config: $config")
      val filesInZip = readFromZip(config.wiki2)

      val trainCorpus =
        filesInZip("wikitext-2/wiki.train.tokens").map(_.toShort)

      val validCorpus =
        filesInZip("wikitext-2/wiki.valid.tokens").map(_.toShort)

      val maxLength = config.maxLength
      Scope.root { implicit scope =>
        scribe.info(
          s"Train corpus length: ${trainCorpus.length} bytes"
        )

        val device =
          if (config.gpus.nonEmpty) CudaDevice(config.gpus.head) else CPU

        val model = allocateModel(device, maxLength)

        val extraModels = config.gpus.drop(1).map { deviceNum =>
          val device = CudaDevice(deviceNum)
          allocateModel(device, maxLength)
        }

        val checkpointedState = config.checkpointLoad.map { state =>
          StateIO
            .readFromFile(new File(state), device)
            .asInstanceOf[SimpleLoopState]
        }

        scribe.info(f"Learnable parameters: ${model.module.learnableParameters}%,d")

        val trainEpochs = (_: IOLoops.TrainingLoopContext) =>
          lamp.data.languagemodel.autoregressiveMinibatchesFromCorpus(
            minibatchSize = config.trainBatchSize,
            numBatches = config.numBatchesPerEpoch,
            corpus = trainCorpus,
            blockLength = maxLength
          )
        val validEpochs = (_: IOLoops.TrainingLoopContext) =>
          lamp.data.languagemodel.autoregressiveMinibatchesFromCorpus(
            minibatchSize = config.trainBatchSize,
            numBatches = config.numBatchesPerEpoch,
            corpus = validCorpus,
            blockLength = maxLength
          )

        val optimizer = RAdam.factory(
          weightDecay = simple(0.00),
          learningRate = simple(config.learningRate),
          clip = Some(1d)
        )

        val (_, trainedModel, _, _, _) = IOLoops
          .epochs(
            model = model,
            optimizerFactory = optimizer,
            trainBatchesOverEpoch = trainEpochs,
            validationBatchesOverEpoch = Some(validEpochs),
            epochs = config.epochs,
            initState = checkpointedState,
            logger = Some(scribe.Logger("training")),
            validationFrequency = 1,
            dataParallelModels = extraModels,
            checkpointState = Some((state: LoopState, _: Unit) =>
              config.checkpointSave
                .map { file =>
                  StateIO.stateToFile(new File(file))(state)
                }
                .getOrElse(IO.unit)
            )
          )
          .unsafeRunSync()
        scribe.info("Training done.")

        println(trainedModel)

      }
    case Some(config) =>
      scribe.info(s"Config: $config")
      scribe.info(s"Inference mode. Extending '${config.extend.get}'")

      val maxLength = config.maxLength
      Scope.root { implicit scope =>
        val device =
          if (config.gpus.nonEmpty) CudaDevice(config.gpus.head) else CPU

        val model = allocateModel(device, maxLength = maxLength).module

        val checkpointedState = config.checkpointLoad
          .map { state =>
            StateIO
              .readFromFile(new File(state), device)
              .asInstanceOf[SimpleLoopState]
          }
          .getOrElse(throw new RuntimeException("Can't load"))

        model.load(checkpointedState.model)

        val modelAsEval = model.languageModel.asEval

        val inferred = lamp.data.languagemodel
          .autoregressiveInference(
            modelAsEval,
            modelBlockSize = maxLength,
            prefix = config.extend.get.getBytes("US-ASCII").map(_.toShort),
            length = 30,
            padToken = 255
          )(scope)
          .unsafeRunSync()

        scribe.info(s"Extended: '${new String(inferred.map(_.toByte))}'")

        ()
      }
    case _ =>
    // arguments are bad, error message will have been displayed
  }
  scribe.info("END")
}
