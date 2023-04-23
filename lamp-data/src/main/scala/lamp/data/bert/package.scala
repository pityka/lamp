package lamp.data

import lamp._
import lamp.nn.bert.{BertPretrainInput, BertLossInput}
import lamp.data.BatchStream.scopeInResource
import lamp.autograd.const
import scala.collection.compat.immutable.ArraySeq

package object bert {

  def pad(v: Array[Int], paddedLength: Int, padElem: Int) = {
    val t = v.++(Array.fill(paddedLength - v.length)(padElem))
    assert(t.length == paddedLength)
    t
  }

  def makeMaskForMaskedLanguageModel(
      bertTokens: Array[Int],
      maximumTokenId: Int,
      clsToken: Int,
      sepToken: Int,
      maskToken: Int,
      rng: scala.util.Random
  ) = {
    val mlmPositions = rng
      .shuffle(
        ArraySeq.unsafeWrapArray(
          Array
            .range(0, bertTokens.length)
            .filter(i => bertTokens(i) != clsToken && bertTokens(i) != sepToken)
        )
      )
      .take(math.max(1, (bertTokens.length * 0.15).toInt))

    val mlmMaskedInput = mlmPositions.map { idx =>
      val target = bertTokens(idx)
      val r = rng.nextDouble()
      val input =
        if (r < 0.8) maskToken
        else if (r < 0.9) rng.nextInt(maximumTokenId)
        else target
      input
    }
    val mlmTarget = mlmPositions.map(bertTokens)

    val mlmPositionsI = mlmPositions.zipWithIndex.toMap
    val maskedBertTokens = bertTokens.zipWithIndex.map { case (token, idx) =>
      if (mlmPositionsI.contains(idx))
        mlmMaskedInput(mlmPositionsI(idx))
      else token
    }

    (mlmPositions.toArray, mlmTarget.toArray, maskedBertTokens)
  }

  def prepareParagraph[S: Sc](
      paragraph: Vector[Array[Int]],
      maximumTokenId: Int,
      clsToken: Int,
      sepToken: Int,
      padToken: Int,
      maskToken: Int,
      maxLength: Int,
      rng: scala.util.Random
  ) = {

    def cpy(v: Array[Int]) = {
      STen.fromLongArray(v.map(_.toLong), List(v.length), CPU)
    }

    val maxNumPredictionPositions = (maxLength * 0.15).toInt
    val numSentences = paragraph.size
    val windowSize = (maxLength - 3) / 2

    def window(sentence: Array[Int]) = {
      if (sentence.length <= windowSize) sentence
      else {
        val maxStart = sentence.size - windowSize
        val start = rng.nextInt(maxStart)
        sentence.slice(start, start + windowSize)
      }
    }

    paragraph.zipWithIndex
      .dropRight(1)
      .map { case (sentence0, sentenceIdx) =>
        val trueNextSentence = rng.nextBoolean()
        val nextSentence0 =
          if (trueNextSentence) paragraph(sentenceIdx + 1)
          else paragraph(rng.nextInt(numSentences))

        val sentence = window(sentence0)
        val nextSentence = window(nextSentence0)

        val bertTokens = Array(clsToken)
          .++(sentence)
          .++(Array(sepToken))
          .++(nextSentence)
          .++(Array(sepToken))

        assert(bertTokens.length <= maxLength)

        val (mlmPositions, mlmTarget, maskedBertTokens) =
          makeMaskForMaskedLanguageModel(
            bertTokens,
            maximumTokenId = maximumTokenId,
            clsToken = clsToken,
            sepToken = sepToken,
            maskToken = maskToken,
            rng = rng
          )

        val bertSegments = Array(0)
          .++(sentence.map(_ => 0))
          .++(Array(0))
          .++(nextSentence.map(_ => 1))
          .++(Array(1))

        (
          trueNextSentence,
          cpy(pad(maskedBertTokens, maxLength, padToken)),
          cpy(pad(bertSegments, maxLength, 0)),
          cpy(pad(mlmPositions, maxNumPredictionPositions, 0)),
          cpy(pad(mlmTarget, maxNumPredictionPositions, padToken)),
          bertTokens.length.toLong
        )

      }

  }

  def minibatchesFromParagraphs(
      minibatchSize: Int,
      dropLast: Boolean,
      paragraphs: Vector[Vector[Array[Int]]], // paragraphs - sentences - tokens
      maximumTokenId: Int,
      clsToken: Int,
      sepToken: Int,
      padToken: Int,
      maskToken: Int,
      maxLength: Int,
      rng: scala.util.Random
  ) = {
    def makeNonEmptyBatch(idx: Array[Int], device: Device) = {
      assert(idx.nonEmpty)
      scopeInResource.map { implicit scope =>
        val (tokens, segments, positions, mlmTargets, nsTargets,tokenMaxLength) = Scope {
          implicit scope =>
            val ps = ArraySeq.unsafeWrapArray(idx).flatMap { paragraphIdx =>
              prepareParagraph(
                paragraphs(paragraphIdx),
                maximumTokenId,
                clsToken,
                sepToken,
                padToken,
                maskToken,
                maxLength,
                rng
              )
            }

            val nextSentenceTarget =
              STen
                .fromLongArray(
                  ps.map(v => if (v._1) 1L else 0L).toArray,
                  List(ps.length),
                  CPU
                )
                .castToFloat
            val maskedBertTokens = STen.stack(dim = 0, tensors = ps.map(_._2))
            val bertSegments = STen.stack(dim = 0, tensors = ps.map(_._3))
            val mlmPositions = STen.stack(dim = 0, tensors = ps.map(_._4))
            val mlmTarget = STen.stack(dim = 0, tensors = ps.map(_._5))
            val tokenMaxLength = STen.fromLongArray(ps.map(_._6).toArray)

            device.withOtherStreamThenSync(synchronizeBefore = false) {

              (
                device.to(maskedBertTokens),
                device.to(bertSegments),
                device.to(mlmPositions),
                device.to(mlmTarget),
                device.to(nextSentenceTarget),
                device.to(tokenMaxLength),
              )
            }
        }

        val batch = BertLossInput(
          input = BertPretrainInput(
            tokens = const(tokens),
            segments = const(segments),
            positions = positions,
            maxLength = Option(tokenMaxLength)
          ),
          maskedLanguageModelTarget = mlmTargets,
          wholeSentenceTarget = nsTargets
        )

        val fakeTarget = STen.zeros(List(minibatchSize), tokens.options)

        NonEmptyBatch((batch, fakeTarget))
      }

    }

    val idx = {
      val t = rng
        .shuffle(ArraySeq.unsafeWrapArray(Array.range(0, paragraphs.length)))
        .grouped(minibatchSize)
        .map(_.toArray)
        .toList
      if (dropLast) t.dropRight(1)
      else t
    }

    BatchStream.fromIndices(idx.toArray)(makeNonEmptyBatch)

  }

}
