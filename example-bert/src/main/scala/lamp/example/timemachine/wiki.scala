package lamp.example.bert

import lamp._
import lamp.nn._
import lamp.data._

// import lamp.CudaDevice
// import lamp.CPU
// import lamp.nn.SupervisedModel
// import lamp.nn.LossFunctions
// import lamp.data.{Reader, Text, IOLoops}
// import java.io.File
// import lamp.nn.AdamW
// import lamp.nn.simple
// import cats.effect.Resource
// import cats.effect.IO
// import lamp.nn.Fun
// import lamp.DoublePrecision
// import lamp.SinglePrecision
import java.nio.charset.CodingErrorAction
import java.nio.charset.Charset
// import scala.io.Codec
// import lamp.nn.Embedding
// import lamp.nn.SeqLinear
// import lamp.nn.LSTM
// import lamp.nn.statefulSequence
// import lamp.Scope
// import lamp.STen
import cats.effect.unsafe.implicits.global
import java.io.FileInputStream
import java.util.zip.ZipInputStream
import lamp.data
import java.io.File

case class CliConfig(
    cuda: Boolean = false,
    wiki2: String = "",
    trainBatchSize: Int = 32,
    validationBatchSize: Int = 32,
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
      opt[Unit]("gpu").action((_, c) => c.copy(cuda = true)),
      opt[String]("wiki2").action((x, c) => c.copy(wiki2 = x)),
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

  val asciiSilentCharsetDecoder = Charset
    .forName("UTF8")
    .newDecoder()
    .onMalformedInput(CodingErrorAction.REPLACE)
    .onUnmappableCharacter(CodingErrorAction.REPLACE)

  def readFromZip(zip: String) = {
    val zis = new ZipInputStream(new FileInputStream(zip));
    var ze = zis.getNextEntry();
    val map = scala.collection.mutable.Map.empty[String, Seq[String]]
    while (ze != null) {
      val name = ze.getName()
      val str = scala.io.Source
        .fromInputStream(zis)(asciiSilentCharsetDecoder)
        .getLines()
        .toSeq
      map += ((name, str))
      ze = zis.getNextEntry

    }
    map.toMap

  }

  def makeParagraphs(
      lines: Seq[String],
      vocab: Map[String, Int],
      unknown: Int
  ) = {
    lines
      .map(_.trim().toLowerCase().split(" \\. "))
      .filter(_.size >= 2)
      .map { paragraph =>
        paragraph.map { sentence =>
          val words = sentence.split(' ')
          words.map(v => vocab.getOrElse(v, unknown)).toArray
        }.toVector
      }
      .toVector
  }

  OParser.parse(parser1, args, CliConfig()) match {
    case Some(config) =>
      scribe.info(s"Config: $config")
      val filesInZip = readFromZip(config.wiki2)
      val (_, vocab) = data.Text.wordsToIntegers(
        filesInZip.values.map(_.mkString(" ")).mkString(" "),
        minimumTokenId = 4,
        minimumFrequency = 100
      )
      val clsToken = 0
      val sepToken = 1
      val padToken = 2
      val maskToken = 3
      val unknownToken = 4

      val maxToken = vocab.values.max
      val trainParagraphs =
        makeParagraphs(
          filesInZip("wikitext-2/wiki.train.tokens"),
          vocab,
          unknownToken
        )
      val validParagraphs =
        makeParagraphs(
          filesInZip("wikitext-2/wiki.valid.tokens"),
          vocab,
          unknownToken
        )
      Scope.root { implicit scope =>
        scribe.info(
          s"Vocabulary size ${vocab.size}, train num sentences: ${trainParagraphs.map(_.size).sum}"
        )

        val device = if (config.cuda) CudaDevice(0) else CPU
        val tensorOptions = device.options(SinglePrecision)

        val maxLength = 32

        val net = lamp.nn.bert.BertLoss.apply(
          maxLength = maxLength,
          vocabularySize = vocab.size + 5,
          segmentVocabularySize = 2,
          mlmHiddenDim = 32,
          wholeStentenceHiddenDim = 32,
          numBlocks = 2,
          embeddingDim = 32,
          attentionHiddenPerHeadDim = 8,
          attentionNumHeads = 4,
          bertEncoderMlpHiddenDim = 32,
          dropout = 0d,
          padToken = padToken,
          tOpt = tensorOptions,
          linearized = false,
          positionEmbedding = None
        )
        config.checkpointLoad
          .foreach { load =>
            scribe.info(s"Loading parameters from file $load")
            data.Reader
              .loadFromFile(net, new File(load), device, false)
              .unsafeRunSync()
          }
        val model = SupervisedModel(net, LossFunctions.Identity)

        scribe.info("Learnable parameters: " + net.learnableParameters)

        val rng = new scala.util.Random
        val trainEpochs = (_: IOLoops.TrainingLoopContext) =>
          lamp.data.bert.minibatchesFromParagraphs(
            minibatchSize = config.trainBatchSize,
            dropLast = true,
            paragraphs = trainParagraphs,
            maximumTokenId = maxToken,
            clsToken = clsToken,
            sepToken = sepToken,
            padToken = padToken,
            maskToken = maskToken,
            maxLength = maxLength,
            rng = rng
          )
        val validEpochs = (_: IOLoops.TrainingLoopContext) =>
          lamp.data.bert.minibatchesFromParagraphs(
            minibatchSize = config.trainBatchSize,
            dropLast = true,
            paragraphs = validParagraphs,
            maximumTokenId = maxToken,
            clsToken = clsToken,
            sepToken = sepToken,
            padToken = padToken,
            maskToken = maskToken,
            maxLength = maxLength,
            rng = rng
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
            epochs = 10,
            logger = Some(scribe.Logger("training")),
            validationFrequency = 1
          )
          .unsafeRunSync()
        scribe.info("Training done.")

        println(trainedModel)

      }
    case _ =>
    // arguments are bad, error message will have been displayed
  }
  scribe.info("END")
}
