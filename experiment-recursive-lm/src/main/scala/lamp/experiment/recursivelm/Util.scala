package lamp.experiment.recursivelm

import lamp._
import java.io.File
import cats.effect.IO
import lamp.data.Codec
import lamp.data.CodecFactory
import lamp.data.IdentityCodec

object Util {

  def prepareCorpora(config: CliConfig)(implicit scope: Scope) = Scope.bracket {
    implicit scope =>
      for {

        rawTrainCorpus <-
          Util.readBytesFromFile(config.trainFile, config.fileMaxLength)

        _ = scribe.info(f"Read raw corpus ${rawTrainCorpus.shape(0)}%,d")

        codec <- Util.readOrTrainCodec(
          config.bpeFile,
          rawTrainCorpus
            .slice(0, 0, 1000000, 1)
            .toByteArray
            .map(b => b.toChar.toLower.toByte),
          Model.codecFactory
        )

                _ = scribe.info(s"Trained codec: $codec")


        trainCorpus <-
          Util.encodeOrReadTokens(
            rawTrainCorpus,
            new File(config.trainFile + ".tokens"),
            codec,
            config.parallelism
          )

        _ = scribe.info(
          s"Train corpus length: ${trainCorpus.numel} tokens"
        )

        validCorpus <-
          (if (config.validFile.isEmpty) IO.pure(Option.empty[STen])
           else
             Util
               .readBytesFromFile(config.validFile, config.fileMaxLength)
               .flatMap(corp =>
                 Util.encodeOrReadTokens(
                   corp,
                   new File(config.validFile + ".tokens"),
                   codec,
                   config.parallelism
                 )
               )
               .map(v => Option(v)))

        _ = scribe.info(
          s"Valid corpus length: ${validCorpus.map(_.numel)} tokens"
        )

      } yield (trainCorpus, validCorpus)
  }

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

  def readBytesFromFile[S: Sc](file: String, maxLength: Long): IO[STen] =
    IO.interruptible {
      val l = math.min(maxLength, new File(file).length())
      STen.fromFile(
        path = file,
        offset = 0,
        length = l,
        scalarTypeByte = 1,
        pin = false
      )

    }
  def saveTokens(file: File, tokens: STen): IO[Unit] = {
    lamp.data.Writer
      .writeTensorsIntoFile(List(tokens), file)
      .map(_.toOption.get)
  }
  def readTokens[S: Sc](file: File): STen = {
    lamp.data.Reader
      .readTensorsFromFile(file, CPU, false)
      .head // int32 signed
  }

  /* Returns int32 tensor */
  def encodeOrReadTokens[S: Sc](
      corpus: STen,
      file: File,
      codec: Codec,
      parallelism: Int
  ): IO[STen] =
    if (file.canRead) {
      IO.blocking {
        scribe.info(s"Reading tokens file $file")
        readTokens(file)
      }
    } else {
      scribe.info(s"Encoding corpus")
      import cats.syntax.all._
      val len = corpus.shape(0)

      val chunkSize = 1024 * 1024L * 10
      IO.parTraverseN(parallelism)((0L until len by chunkSize).toList) {
        start =>
          IO.interruptible {
            val slice = corpus
              .slice(0, start, math.min(start + chunkSize, len), 1)
              .toByteArray
              .map(b => b.toChar.toLower.toByte)
            val enc = codec.encode(slice).map(_.toInt)
            STen.fromIntArray(enc, List(enc.length), CPU)
          }
      }.flatMap { list =>
        val encoded = STen.cat(list, dim = 0)

        if (codec != IdentityCodec) {
          scribe.info(s"Saving tokens into $file")
          saveTokens(file, encoded).map(_ => encoded)
        } else IO.pure(encoded)
      }
    }

}
