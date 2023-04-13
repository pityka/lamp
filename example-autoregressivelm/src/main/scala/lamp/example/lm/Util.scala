package lamp.example.lm

import java.nio.charset.CodingErrorAction
import java.nio.charset.Charset
import lamp._
import java.io.FileInputStream
import java.io.File
import cats.effect.IO
import lamp.data.Codec
import lamp.data.CodecFactory

object Util {

  def prepareCorpora(config: CliConfig) = for {

    rawTrainCorpus <-
      Util.readBytesFromFile(config.trainFile, config.fileMaxLength)

    _ = scribe.info(f"Read raw corpus ${rawTrainCorpus.length}%,d")

    codec <- Util.readOrTrainCodec(
      config.bpeFile,
      rawTrainCorpus.take(300000),
      Model.codecFactory
    )

    trainCorpus <- IO(
      Util.encodeOrReadTokens(
        rawTrainCorpus,
        new File(config.trainFile + ".tokens"),
        codec
      )
    )

    _ = scribe.info(
      s"Train corpus length: ${trainCorpus.length} tokens"
    )

    validCorpus <-
      Util
        .readBytesFromFile(config.validFile, config.fileMaxLength)
        .map(corp =>
          Util.encodeOrReadTokens(
            corp,
            new File(config.validFile + ".tokens"),
            codec
          )
        )

    _ = scribe.info(
      s"Valid corpus length: ${validCorpus.length} tokens"
    )

  } yield (trainCorpus, validCorpus)

  def readOrTrainCodec[T <: Codec](
      bpeFile: Option[File],
      trainData: Array[Byte],
      codecFactory: CodecFactory[T]
  ): IO[T] = {

    if (bpeFile.isDefined && bpeFile.get.canRead)
      codecFactory.readFromFile(bpeFile.get)
    else
      IO {
        val bpe = codecFactory.train(
          corpus = trainData
        )
        bpeFile
          .map { file =>
            bpe
              .saveToFile(file)
              .map(_ => bpe)
          }
          .getOrElse(IO.pure(bpe))
      }.flatten
  }

  def readBytesFromFile(file: String, maxLength: Int): IO[Array[Byte]] =
    IO.blocking {
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
  def saveTokens(file: File, array: => Array[Char]): IO[Unit] = {
    Scope.root { implicit scope =>
      val t = STen.fromShortArray(array.map(_.toShort), List(array.size), CPU)
      lamp.data.Writer
        .writeTensorsIntoFile(List(t), file)
        .map(_.toOption.get)
    }
  }
  def readTokens(file: File): Array[Char] = {
    Scope.unsafe { implicit scope =>
      lamp.data.Reader
        .readTensorsFromFile(file, CPU, false)
        .head
        .toShortArray
        .map(_.toChar)
    }
  }

  def encodeOrReadTokens(
      corpus: Array[Byte],
      file: File,
      codec: Codec
  ): Array[Char] =
    if (file.canRead) {
      scribe.info(s"Reading tokens file $file")
      readTokens(file)
    } else {
      scribe.info(s"Encoding corpus")
      val enc = codec.encode(corpus)
      scribe.info(s"Saving tokens into $file")
      saveTokens(file, enc)
      enc
    }

}
