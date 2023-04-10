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
import lamp.data.bytesegmentencoding.ByteSegmentCodecFactory

case class CliConfig(
    gpus: Seq[Int] = Nil,
    wiki2: String = "",
    trainBatchSize: Int = 64,
    epochs: Int = 2000,
    learningRate: Double = 0.0001,
    weightDecay: Double = 0.0,
    samplingTemperature: Double = 1.0,
    dropout: Double = 0.0,
    numBatchesPerEpoch: Int = 100,
    maxLength: Int = 128,
    checkpointSave: Option[String] = None,
    checkpointLoad: Option[String] = None,
    extend: Option[String] = None,
    extendLength: Int = 50
)

object Train extends App {
  scribe.info("Logger start")

  val vocabularySize = 512

  val codecFactory = ByteSegmentCodecFactory(
    vocabularyMin = 1,
    vocabularyMax = (vocabularySize - 1).toChar,
    maxMergedSegmentLength = 3,
    unknownToken = 0.toChar,
    unknownByte = '?'.toByte
  )

  def allocateModel(device: Device, maxLength: Int)(implicit scope: Scope) = {
    val tensorOptions = device.options(SinglePrecision)
    val embeddingDim = 512
    val layers = 6
    val numHeads = 16
    val net = lamp.nn.languagemodel.LanguageModelLoss.apply(
      maxLength = maxLength,
      vocabularySize = vocabularySize,
      numBlocks = layers,
      embeddingDim = embeddingDim,
      attentionHiddenPerHeadDim = embeddingDim / numHeads,
      attentionNumHeads = numHeads,
      encoderMlpHiddenDim = embeddingDim * 4,
      dropout = 0d,
      padToken = -1000L,
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
      opt[String]("extend").action((x, c) => c.copy(extend = Some(x))),
      opt[Int]("extend-length").action((x, c) => c.copy(extendLength = x)),
      opt[Int]("epochs").action((x, c) => c.copy(epochs = x)),
      opt[Int]("batches-per-epoch").action((x, c) =>
        c.copy(numBatchesPerEpoch = x)
      ),
      opt[Int]("max-length").action((x, c) => c.copy(maxLength = x)),
      opt[Double]("learning-rate").action((x, c) => c.copy(learningRate = x)),
      opt[Double]("weight-decay").action((x, c) => c.copy(weightDecay = x)),
      opt[Double]("dropout").action((x, c) => c.copy(dropout = x)),
      opt[Double]("sampling-temperature").action((x, c) =>
        c.copy(samplingTemperature = x)
      ),
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
        .map(_.toLowerCase())
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
      val bpeFile = config.checkpointLoad.map(file =>
        new File(file + ".bytesegmentencoding.json")
      )
      val filesInZip = readFromZip(config.wiki2)

      val rawTrainCorpus =
        filesInZip("wikitext-2/wiki.train.tokens")

      val codec =
        if (bpeFile.isDefined && bpeFile.get.canRead)
          codecFactory.readFromFile(bpeFile.get)
        else {
          val bpe = codecFactory.train(
            corpus = rawTrainCorpus.take(200000)
          )
          config.checkpointSave.foreach { file =>
            bpe.saveToFile(new File(file + ".bytesegmentencoding.json"))

          }
          bpe
        }

      // scribe.info(
      //   s"Trained encoding. Kmers: \n ${bpe.trainedEncoding
      //     .map { case (pattern, sub) =>
      //       new String(pattern.toArray) -> sub.toInt
      //     }
      //     .mkString("\n")}"
      // )

      val trainCorpus = codec.encode(rawTrainCorpus)

      scribe.info(
        s"Train corpus length: ${trainCorpus.length} bytes"
      )

      val validCorpus =
        codec.encode(
          filesInZip("wikitext-2/wiki.valid.tokens")
        )

      scribe.info(
        s"Valid corpus length: ${validCorpus.length} bytes"
      )

      val maxLength = config.maxLength
      Scope.root { implicit scope =>
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

        scribe.info(
          f"Learnable parameters: ${model.module.learnableParameters}%,d"
        )

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

        val optimizer = AdamW.factory(
          weightDecay = simple(config.weightDecay),
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
      val bpeFile = config.checkpointLoad.map(file =>
        new File(file + ".bytesegmentencoding.json")
      )
      val codec = codecFactory.readFromFile(bpeFile.get)

      val maxLength = config.maxLength
      Scope.root { implicit scope =>
        val device = CPU

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

        val rawPrefix = config.extend.get.getBytes("US-ASCII")

        val encodedPrefix = codec.encode(rawPrefix)

        val inferred = lamp.data.languagemodel
          .autoregressiveInference(
            modelAsEval,
            modelBlockSize = maxLength,
            prefix = encodedPrefix,
            length = config.extendLength,
            temperature = config.samplingTemperature
          )(scope)
          .unsafeRunSync()

        println(inferred.map(_.toInt).toVector)

        val decoded = codec.decode(inferred)

        scribe.info(s"Extended: '${new String(decoded)}'")

        ()
      }
    case _ =>
    // arguments are bad, error message will have been displayed
  }
  scribe.info("END")
}
