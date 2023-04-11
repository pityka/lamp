package lamp.example.timemachine

import lamp._
import lamp.nn._
import lamp.data._

import java.nio.charset.CodingErrorAction
import java.nio.charset.Charset
import cats.effect.unsafe.implicits.global
import java.io.FileInputStream
import java.io.File
import cats.effect.IO
import lamp.data.bytesegmentencoding.ByteSegmentCodecFactory
import lamp.data.bytesegmentencoding.ByteSegmentCodec

case class CliConfig(
    gpus: Seq[Int] = Nil,
    trainFile: String = "",
    validFile: String = "",
    fileMaxLength: Int = Int.MaxValue - 100,
    trainBatchSize: Int = 12,
    epochs: Int = 10000,
    learningRate: Double = 0.0001,
    weightDecay: Double = 0.0,
    samplingTemperature: Double = 1.0,
    dropout: Double = 0.0,
    numBatchesPerEpoch: Int = 100,
    checkpointSave: Option[String] = None,
    checkpointLoad: Option[String] = None,
    extend: Option[String] = None,
    extendLength: Int = 50,
    gradientAccumSteps: Int = 5,
    // config for distributed training
    distributed: Boolean = false,
    gpu: Int = 0,
    rank: Int = 0,
    nranks: Int = 0,
    rootAddress: String = "",
    rootPort: Int = 28888,
    myAddress: String = "",
    myPort: Int = 28888
)

object Train extends App {
  scribe.info("Logger start")

  val vocabularySize = 2048

  val contextLength = 256

  val codecFactory = ByteSegmentCodecFactory(
    vocabularyMin = 1,
    vocabularyMax = (vocabularySize - 1).toChar,
    maxMergedSegmentLength = 5,
    unknownToken = 0.toChar,
    unknownByte = '?'.toByte
  )

  def allocateModel(device: Device)(implicit scope: Scope) = {
    val tensorOptions = device.options(SinglePrecision)
    val embeddingDim = 768
    val layers = 12
    val numHeads = 12
    val net = lamp.nn.languagemodel.LanguageModelLoss.apply(
      maxLength = contextLength,
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
    scribe.info(
      s"Allocated model on $device . embedding=$embeddingDim layers=$layers num-heads=$numHeads"
    )
    SupervisedModel(net, LossFunctions.Identity)
  }

  import scopt.OParser
  val builder = OParser.builder[CliConfig]
  val parser1 = {
    import builder._
    OParser.sequence(
      programName("languagemodel example"),
      head("trains an autoregressive language model"),
      opt[Seq[Int]]("gpus")
        .action((x, c) => c.copy(gpus = x))
        .text("list of gpus or empty for cpu"),
      opt[String]("train-file")
        .action((x, c) => c.copy(trainFile = x))
        .text("file containing ascii bytes"),
      opt[String]("valid-file")
        .action((x, c) => c.copy(validFile = x))
        .text("file containing ascii bytes"),
      opt[Int]("batch-size").action((x, c) => c.copy(trainBatchSize = x)),
      opt[String]("extend")
        .action((x, c) => c.copy(extend = Some(x)))
        .text("Turns on inference model. Extend this text in inference mode"),
      opt[Int]("extend-length")
        .action((x, c) => c.copy(extendLength = x))
        .text("extend this number of tkens in inference model"),
      opt[Int]("train-file-max-length").action((x, c) =>
        c.copy(fileMaxLength = x)
      ),
      opt[Int]("epochs").action((x, c) => c.copy(epochs = x)),
      opt[Int]("gradient-accum-steps").action((x, c) =>
        c.copy(gradientAccumSteps = x)
      ),
      opt[Int]("batches-per-epoch").action((x, c) =>
        c.copy(numBatchesPerEpoch = x)
      ),
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

  def readFromFile(file: String, maxLength: Int): Array[Byte] = {
    val zis = new FileInputStream(file)

    val buffer = zis.readNBytes(maxLength)
    val b2 = Array.ofDim[Byte](buffer.length)
    var i = 0
    while (i < b2.length) {
      b2(i) = buffer(i).toChar.toLower.toByte
      i += 1
    }
    b2
  }

  OParser.parse(parser1, args, CliConfig()) match {
    case Some(config) if config.extend.isEmpty =>
      scribe.info(s"Config: $config")
      val bpeFile = config.checkpointLoad.map(file =>
        new File(file + ".bytesegmentencoding.json")
      )

      val rawTrainCorpus =
        readFromFile(config.trainFile, config.fileMaxLength)

      scribe.info(f"Read raw corpus ${rawTrainCorpus.length}%,d")

      val codec =
        if (bpeFile.isDefined && bpeFile.get.canRead)
          codecFactory.readFromFile(bpeFile.get)
        else {
          val bpe = codecFactory.train(
            corpus = rawTrainCorpus.take(300000)
          )
          config.checkpointSave.foreach { file =>
            bpe.saveToFile(new File(file + ".bytesegmentencoding.json"))

          }
          bpe
        }

      scribe.info(
        s"Trained encoding. Kmers: \n ${codec
          .asInstanceOf[ByteSegmentCodec]
          .trained
          .map { case (pattern, sub) =>
            new String(pattern.toArray) -> sub.toInt
          }
          .mkString("\n")}"
      )

      val trainCorpus = codec.encode(rawTrainCorpus)

      scribe.info(
        s"Train corpus length: ${trainCorpus.length} tokens"
      )

      val validCorpus =
        codec.encode(
          readFromFile(config.validFile, config.fileMaxLength)
        )

      scribe.info(
        s"Valid corpus length: ${validCorpus.length} tokens"
      )

      if (config.distributed) {
        Scope.root { implicit scope =>
          scribe.info(
            s"Distributed training rank: ${config.rank} nrank:${config.nranks} gpu:${config.gpu}"
          )
          val device = CudaDevice(config.gpu)
          val model = allocateModel(device)

          val actorSystem = akka.actor.ActorSystem(
            name = s"lm-${config.rank}",
            config = Some(
              com.typesafe.config.ConfigFactory.parseString(
                s"""
akka {
  actor {
    provider = remote 
  }
  remote {
    artery {
      transport = tcp 
      canonical.hostname = "${config.myAddress}"
      canonical.port = ${config.myPort}
    }
  }
}
                """
              )
            )
          )

          val trainEpochs = () =>
            lamp.data.languagemodel
              .autoregressiveMinibatchesFromCorpus(
                minibatchSize = config.trainBatchSize,
                numBatches = config.numBatchesPerEpoch,
                corpus = trainCorpus,
                blockLength = contextLength
              )
              .withoutEmptyBatches
              .everyNth(n = config.nranks, offset = config.rank)

          val validEpochs = () =>
            lamp.data.languagemodel
              .autoregressiveMinibatchesFromCorpus(
                minibatchSize = config.trainBatchSize,
                numBatches = config.numBatchesPerEpoch,
                corpus = validCorpus,
                blockLength = contextLength
              )
              .withoutEmptyBatches
              .everyNth(n = config.nranks, offset = config.rank)

          val program = if (config.rank == 0) {

            val optimizer = AdamW.factory(
              weightDecay = simple(config.weightDecay),
              learningRate = simple(config.learningRate),
              clip = Some(1d)
            )

            // val checkpointedState = config.checkpointLoad.map { state =>
            //   StateIO
            //     .readFromFile(new File(state), device)
            //     .asInstanceOf[SimpleLoopState]
            // }

            val comm = new lamp.distributed.akka.AkkaCommunicationServer(
              actorSystem
            )
            lamp.data.distributed.driveDistributedTraining(
              nranks = config.nranks,
              gpu = config.gpu,
              controlCommunication = comm,
              model = model,
              optimizerFactory = optimizer,
              trainBatches = trainEpochs,
              validationBatches = validEpochs,
              maxEpochs = config.epochs
              // initState = checkpointedState,
              // checkpointState = Some((state: LoopState, _: Unit) =>
              //   config.checkpointSave
              //     .map { file =>
              //       StateIO.stateToFile(new File(file))(state)
              //     }
              //     .getOrElse(IO.unit)
              // )
            )

          } else {
            import scala.concurrent.duration._
            val comm = new lamp.distributed.akka.AkkaCommunicationClient(
              actorSystem,
              config.rootAddress,
              config.rootPort,
              "cifar-0",
              600 seconds
            )

            lamp.data.distributed.followDistributedTraining(
              rank = config.rank,
              nranks = config.nranks,
              gpu = config.gpu,
              controlCommunication = comm,
              model = model,
              trainBatches = trainEpochs,
              validationBatches = validEpochs
            )

          }

          program.unsafeRunSync()
          actorSystem.terminate()
          ()
        }

      } else {

        Scope.root { implicit scope =>
          val device =
            if (config.gpus.nonEmpty) CudaDevice(config.gpus.head) else CPU

          val model = allocateModel(device)

          val extraModels = config.gpus.drop(1).map { deviceNum =>
            val device = CudaDevice(deviceNum)
            allocateModel(device)
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
              blockLength = contextLength
            )
          val validEpochs = (_: IOLoops.TrainingLoopContext) =>
            lamp.data.languagemodel.autoregressiveMinibatchesFromCorpus(
              minibatchSize = config.trainBatchSize,
              numBatches = config.numBatchesPerEpoch,
              corpus = validCorpus,
              blockLength = contextLength
            )

          val optimizer = AdamW.factory(
            weightDecay = simple(config.weightDecay),
            learningRate = simple(config.learningRate),
            clip = Some(1d)
          )

          val (_, _, _, _, _) = IOLoops
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
              accumulateGradientOverNBatches = config.gradientAccumSteps,
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

        }

      }
    case Some(config) =>
      scribe.info(s"Config: $config")
      scribe.info(s"Inference mode. Extending '${config.extend.get}'")
      val bpeFile = config.checkpointLoad.map(file =>
        new File(file + ".bytesegmentencoding.json")
      )
      val codec = codecFactory.readFromFile(bpeFile.get)

      Scope.root { implicit scope =>
        val device =
          if (config.gpus.nonEmpty) CudaDevice(config.gpus.head) else CPU

        val model = allocateModel(device).module

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
            modelBlockSize = contextLength,
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
