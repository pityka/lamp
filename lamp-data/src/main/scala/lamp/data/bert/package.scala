package lamp.data

import lamp._
import lamp.nn.bert.{BertPretrainInput, BertLossInput}
import lamp.data.BatchStream.scopeInResource
import lamp.autograd.const

package object bert {

  def pad(v: Array[Int], paddedLength: Int, padElem: Int) = {
    val t = v.concat(Array.fill(paddedLength - v.length)(padElem))
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
        Array
          .range(0, bertTokens.length)
          .filter(i => bertTokens(i) != clsToken && bertTokens(i) != sepToken)
          
        
      )
      .take(math.max(1, (bertTokens.length * 0.15).toInt))

    val mlmMaskedInput = mlmPositions.map { idx =>
      val target = bertTokens(idx)
      val r = rng.nextDouble()
      val input =
        if (r < 0.8) maskToken
        else if (r < 0.9) rng.nextInt( maximumTokenId)
        else target
      input
    }
    val mlmTarget = mlmPositions.map(bertTokens)

    val mlmPositionsI = mlmPositions.zipWithIndex.toMap
    val maskedBertTokens = bertTokens.zipWithIndex.map{ case (token, idx) =>
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

    val maxNumPredictionPositions = (maxLength * 0.15).toInt
    val numSentences = paragraph.size

    paragraph.zipWithIndex
      .dropRight(1)
      .map { case (sentence, sentenceIdx) =>
        val trueNextSentence = rng.nextBoolean()
        val nextSentence =
          if (trueNextSentence) paragraph(sentenceIdx + 1)
          else paragraph(rng.nextInt(numSentences))

        if (sentence.length + nextSentence.length + 3 > maxLength) None
        else {

          val bertTokens = Array(clsToken)
            .concat(sentence)
            .concat(Array(sepToken))
            .concat(nextSentence)
            .concat(Array(sepToken))

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
            .concat(sentence.map(_ => 0))
            .concat(Array(0))
            .concat(nextSentence.map(_ => 1))
            .concat(Array(1))

          def to(v: Array[Int]) = STen.fromLongArray(v.map(_.toLong), List(v.length), CPU)

          Some(
            (
              trueNextSentence,
              to(pad(maskedBertTokens, maxLength, padToken)),
              to(pad(bertSegments, maxLength, 0)),
              to(pad(mlmPositions, maxNumPredictionPositions, 0)),
              to(pad(mlmTarget, maxNumPredictionPositions, padToken))
            )
          )
        }

      }
      .filter(_.isDefined)
      .map(_.get)

  }
  def prepareFullDatasetFromTokenizedParagraphs[S: Sc](
      paragraphs: Vector[Vector[Array[Int]]],
      maximumTokenId: Int,
      clsToken: Int,
      sepToken: Int,
      padToken: Int,
      maskToken: Int,
      maxLength: Int,
      rng: scala.util.Random
  ) = {
    val ps = paragraphs.flatMap { paragraph =>
      prepareParagraph(
        paragraph = paragraph,
        maximumTokenId = maximumTokenId,
        clsToken = clsToken,
        sepToken = sepToken,
        padToken = padToken,
        maskToken = maskToken,
        maxLength = maxLength,
        rng = rng
      )
    }
    val nextSentenceTarget =
      STen.fromLongArray(ps.map(v => if (v._1) 1L else 0L).toArray, List(ps.length),CPU).castToFloat
    val maskedBertTokens = STen.stack(dim = 0, tensors = ps.map(_._2))
    val bertSegments = STen.stack(dim = 0, tensors = ps.map(_._3))
    val mlmPositions = STen.stack(dim = 0, tensors = ps.map(_._4))
    val mlmTarget = STen.stack(dim = 0, tensors = ps.map(_._5))

    BertData(
      maskedTokens = maskedBertTokens,
      segments = bertSegments,
      predictionPositions = mlmPositions,
      maskedLanguageModelTarget = mlmTarget,
      nextSentenceTarget = nextSentenceTarget
    )

  }

  def minibatchesFromFull(
      minibatchSize: Int,
      dropLast: Boolean,
      fullData: BertData,
      rng: scala.util.Random
  ) = {
    def makeNonEmptyBatch(idx: Array[Int], device: Device) = {
      scopeInResource.map { implicit scope =>
        val (tokens, segments, positions, mlmTargets, nsTargets) = Scope {
          implicit scope =>
            val idxT = STen.fromLongArray(
              idx.map(_.toLong),
              List(idx.length),
              fullData.maskedTokens.device
            )

            val tokens = fullData.maskedTokens.index(idxT)
            val segments = fullData.segments.index(idxT)
            val positions = fullData.predictionPositions.index(idxT)
            val mlmTargets = fullData.maskedLanguageModelTarget.index(idxT)
            val nsTargets = fullData.nextSentenceTarget.index(idxT)

            device.withOtherStreamThenSync(synchronizeBefore = false) {

              (
                device.to(tokens),
                device.to(segments),
                device.to(positions),
                device.to(mlmTargets),
                device.to(nsTargets)
              )
            }
        }

        val batch = BertLossInput(
          input = BertPretrainInput(
            tokens = const(tokens),
            segments = const(segments),
            positions = positions
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
        .shuffle(Array.range(0, fullData.maskedTokens.shape(0).toInt))
        .grouped(minibatchSize)
        .map(_.toArray)
        .toList
      if (dropLast) t.dropRight(1)
      else t
    }

    BatchStream.fromIndices(idx.toArray)(makeNonEmptyBatch)

  }

}
