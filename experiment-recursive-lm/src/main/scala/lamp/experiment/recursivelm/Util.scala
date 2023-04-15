package lamp.experiment.recursivelm

import lamp._
import java.io.File
import cats.effect.IO
import lamp.data.Codec
import lamp.data.CodecFactory
import java.io.FileInputStream

object Util {

  def prepareDocuments(config: CliConfig)(implicit scope: Scope) =
    Scope.bracket { implicit scope =>
      for {

        rawTrainDocuments <-
          Util.readDocumentBytesFromFile(config.trainFile, config.fileMaxLength)

        _ = scribe.info(
          f"Read raw corpus ${rawTrainDocuments.map(_.shape(0)).sum}%,d"
        )

        codec <- Util.readOrTrainCodec(
          config.bpeFile,
          STen
            .cat(rawTrainDocuments.take(1000).toVector, dim = 0)
            .slice(0, 0, 300000, 1)
            .toByteArray,
          Model.codecFactory
        )

        trainCorpus <-
          Util.encodeOrReadTokens(
            rawTrainDocuments,
            new File(config.trainFile + ".tokens"),
            codec,
            config.parallelism
          )

        _ = scribe.info(
          s"Train corpus length: ${trainCorpus.map(_.shape(0)).sum} tokens"
        )

        validCorpus <-
          Util
            .readDocumentBytesFromFile(config.validFile, config.fileMaxLength)
            .flatMap(docs =>
              Util.encodeOrReadTokens(
                docs,
                new File(config.validFile + ".tokens"),
                codec,
                config.parallelism
              )
            )

        _ = scribe.info(
          s"Valid corpus length: ${validCorpus.map(_.shape(0)).sum} tokens"
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

  case class PileSchema(text: String)
  object PileSchema {
    import com.github.plokhotnyuk.jsoniter_scala.macros._
    import com.github.plokhotnyuk.jsoniter_scala.core._
    implicit val codec: JsonValueCodec[PileSchema] = JsonCodecMaker.make

  }

  def readDocumentBytesFromFile[S: Sc](
      file: String,
      maxLength: Long
  ): IO[Array[STen]] =
    IO.blocking {

      val is = new FileInputStream(new File(file))
      val buffer = org.saddle.Buffer.empty[STen]
      var t = 0L
      try {
        com.github.plokhotnyuk.jsoniter_scala.core
          .scanJsonValuesFromStream[PileSchema](is) { pileSchema =>
            val bytes = pileSchema.text.getBytes("US-ASCII")
            t += bytes.length
            buffer.+=(STen.fromByteArray(bytes, dim = List(bytes.length), CPU))
            t < maxLength
          }
        buffer.toArray
      } finally {
        is.close
      }
     

    }
  def saveTokens(file: File, tokens: Seq[STen]): IO[Unit] = {
    lamp.data.Writer
      .writeTensorsIntoFile(tokens, file)
      .map(_.toOption.get)
  }
  def readTokens[S: Sc](file: File): Array[STen] = {
    lamp.data.Reader
      .readTensorsFromFile(file, CPU, false)
      .toArray // int32 signed
  }

  /* Returns int32 tensor */
  def encodeOrReadTokens[S: Sc](
      documents: Array[STen],
      file: File,
      codec: Codec,
      parallelism: Int
  ): IO[Array[STen]] =
    if (file.canRead) {
      IO.blocking {
        scribe.info(s"Reading tokens file $file")
        readTokens(file)

      }
    } else {
      scribe.info(s"Encoding corpus")
      import cats.syntax.all._

      IO.parTraverseN(parallelism)(documents.toList.filter(_.shape(0) > 1024)) { document =>
        IO.blocking {

          val enc = codec.encode(document.toByteArray).map(_.toInt)
          STen.fromIntArray(enc, List(enc.length), CPU)
        }
      }.flatMap { list =>
        scribe.info(s"Saving tokens into $file")
        saveTokens(file, list).map(_ => list.toArray)
      }
    }

}
