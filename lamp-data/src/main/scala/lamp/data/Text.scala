package lamp.data

import lamp.autograd.{const}
import lamp.TensorHelpers
import aten.ATen
import aten.Tensor
import lamp.Device
import lamp.autograd.Variable
import lamp.nn.StatefulModule
import lamp.nn.InitState
import lamp.nn.FreeRunningRNN
import lamp.Scope
import lamp.STen
import scala.collection.compat.immutable.ArraySeq
import lamp.nn.InitStateSyntax

object Text {
  def sequencePrediction[T, M <: StatefulModule[Variable, Variable, T]](
      batch: Seq[Vector[Long]],
      device: Device,
      module: M with StatefulModule[Variable, Variable, T],
      steps: Int
  )(implicit
      is: InitState[M, T],
      scope: Scope
  ): STen = {
    Scope { implicit scope =>
      val predictionBatch = makePredictionBatch(batch, device)

      FreeRunningRNN(module, steps)
        .forward(predictionBatch -> module.initState)
        ._1
        .argmax(2, false)
        .value

    }
  }
  def sequencePredictionBeam[T, M <: StatefulModule[Variable, Variable, T]](
      prefix: Vector[Long],
      device: Device,
      module: M with StatefulModule[Variable, Variable, T],
      steps: Int,
      startSequence: Int,
      endOfSequence: Int
  )(implicit
      is: InitState[M, T],
      scope: Scope
  ): Seq[(STen, Double)] = {
    val k = 3

    Scope { implicit scope =>
      val predictionBatch = makePredictionBatch(Vector(prefix), device)

      def loop(
          n: Int,
          buffers: Seq[(Seq[(Variable, T, Int)], Double)]
      ): Seq[(Seq[Variable], Double)] = {

        if (n == 0) {
          buffers.map(b => (b._1.map(_._1), b._2))
        } else {
          val candidates = buffers.flatMap { case (sequence, logProb0) =>
            val (lastOutput, lastState, lastToken) = sequence.last
            if (lastToken == endOfSequence) {
              List(
                (
                  sequence,
                  lastOutput,
                  logProb0,
                  lastState,
                  lastToken
                )
              )
            } else {
              val (output, state) =
                module.forward((lastOutput, lastState))

              val lastChar = if (output.shape(0) > 1) {
                val lastTimeStep1 =
                  output.select(0, output.shape(0) - 1)

                lastTimeStep1.view((1L :: lastTimeStep1.shape))

              } else output

              (0 until lastChar.shape(2).toInt).map { i =>
                val selected = lastChar.select(2L, i.toLong)
                val tmp =
                  Tensor.scalarLong(i.toLong, selected.options.toLong.value)
                val index = ATen._unsafe_view(tmp, Array(1L, 1L))
                tmp.release
                val logProb = selected.toDoubleArray.apply(0)
                (
                  sequence,
                  selected.assign(const(STen.owned(index))),
                  logProb + logProb0,
                  state,
                  i
                )
              }
            }

          }
          val (chosen, _) = candidates.sortBy(_._3).reverse.splitAt(k)
          val newBuffers = chosen.map {
            case (sequence, selected, logProb, state, i) =>
              (sequence :+ ((selected, state, i)), logProb)
          }

          loop(
            n - 1,
            newBuffers
          )
        }
      }

      val ret = loop(
        steps,
        Seq(Seq((predictionBatch, module.initState, startSequence)) -> 0d)
      ).sortBy(_._2)
        .reverse
        .map { case (seq, logProb) =>
          val catted = Variable
            .concatenateAddNewDim(
              seq.map(v => v.select(0, v.shape(0) - 1))
            )
            .view(List(seq.size))

          (catted, logProb)
        }

      ret.map(v => (v._1.value.cloneTensor, v._2))

    }
  }

  /** Convert back to text. Tensor shape: time x batch x dim
    */
  def convertLogitsToText(
      tensor: STen,
      vocab: Map[Int, Char]
  )(implicit scope: Scope): Seq[String] = Scope { implicit scope =>
    convertIntegersToText(tensor.argmax(2, false), vocab)
  }

  /** Convert back to text. Tensor shape: time x batch
    */
  def convertIntegersToText(
      tensor: STen,
      vocab: Map[Int, Char]
  ): Seq[String] = {
    Scope.root { implicit scope =>
      val r = tensor.t.toLongArray
      r.grouped(tensor.shape(1).toInt)
        .toVector
        .map(v => v.toSeq.map(l => vocab(l.toInt)).mkString)
    }
  }

  def charsToIntegers(text: String): (Map[Char, Int], Vector[Int]) = {
    val chars = text.toSeq
      .groupBy(identity)
      .toSeq
      .map { case (char, group) =>
        (char, group.size)
      }
      .sortBy(_._2)
      .reverse
      .map(_._1)
      .zipWithIndex
      .toMap

    (chars, text.map(c => chars(c)).toVector)
  }
  def wordsToIntegers(
      text: String,
      minimumTokenId: Int,
      minimumFrequency: Int
  ): (Array[Int], Map[String, Int]) = {
    val words = text
      .split("\\s+")

    val vocab = words
      .groupBy(identity)
      .toSeq
      .map { case (word, repeats) =>
        (word, repeats.size)
      }
      .filter(_._2 >= minimumFrequency)
      .sortBy(_._2)
      .reverse
      .map(_._1)
      .zipWithIndex
      .map(v => (v._1, v._2 + minimumTokenId + 1))
      .toMap

    (words.map(w => vocab.getOrElse(w, minimumTokenId)), vocab)
  }

  def charsToIntegers(text: String, chars: Map[Char, Int]): Vector[Int] = {

    text.map(c => chars(c)).toVector
  }

  def makePredictionBatch(
      examples: Seq[Vector[Long]],
      device: Device
  )(implicit scope: Scope): Variable = {
    val tensor = Scope { implicit scope =>
      val flattenedFeature =
        STen.fromLongArray(examples.flatMap(identity).toArray)
      val viewedFeature =
        flattenedFeature.view(
          examples.size.toLong,
          examples.head.size.toLong
        )
      val transposedFeatures = viewedFeature.transpose(0, 1)
      device.to(transposedFeatures)

    }
    const(tensor)
  }

  /** Yields tensors of shape (time step x batch size)
    */
  def minibatchesFromText(
      text: Vector[Int],
      minibatchSize: Int,
      timeSteps: Int,
      rng: scala.util.Random
  ) = {
    def makeNonEmptyBatch(idx: Array[Int], device: Device) =
      BatchStream.scopeInResource.map { implicit scope =>
        val pairs = idx.map { i =>
          val segmentFeature =
            text.drop(i).take(timeSteps).map(_.toLong).toArray
          val segmentTarget =
            text.drop(i + 1).take(timeSteps).map(_.toLong).toArray
          assert(segmentFeature.length == timeSteps)

          (segmentFeature, segmentTarget)

        }

        val flattenedFeature =
          TensorHelpers
            .fromLongArray(pairs.flatMap(_._1).toArray, device)
        val flattenedTarget =
          TensorHelpers.fromLongArray(pairs.flatMap(_._2).toArray, device)
        val viewedFeature = ATen._unsafe_view(
          flattenedFeature,
          Array(idx.size.toLong, timeSteps.toLong)
        )
        val viewedTarget = ATen._unsafe_view(
          flattenedTarget,
          Array(idx.size.toLong, timeSteps.toLong)
        )
        val transposedFeatures =
          ATen.transpose(viewedFeature, 0, 1)
        val transposedTarget = ATen.transpose(viewedTarget, 0, 1)

        Tensor.releaseAll(
          Array(
            viewedTarget,
            viewedFeature,
            flattenedTarget,
            flattenedFeature
          )
        )

        StreamControl(
          (const(STen.owned(transposedFeatures)), STen.owned(transposedTarget))
        )

      }

    val dropped = text.drop(scala.util.Random.nextInt(timeSteps))
    val numSamples = (dropped.size - 1) / timeSteps
    val idx = rng
      .shuffle(
        ArraySeq.unsafeWrapArray(
          Array.range(0, numSamples * timeSteps, timeSteps)
        )
      )
      .grouped(minibatchSize)
      .toList
      .map(_.toArray)
      .dropRight(1)

    scribe.info(
      s"Total batches: ${idx.size}. Each $timeSteps token long and has $minibatchSize examples."
    )
    assert(idx.forall(_.size == minibatchSize))
    BatchStream.fromIndices(idx.toArray)(makeNonEmptyBatch)

  }

  def sentenceToPaddedVec(
      sentence: String,
      maxLength: Int,
      pad: Int,
      vocabulary: Map[Char, Int]
  ): Array[Int] =
    sentence.map(vocabulary).toArray.take(maxLength).padTo(maxLength, pad)

  def sentencesToPaddedMatrix(
      sentences: Seq[String],
      maxLength: Int,
      pad: Int,
      vocabulary: Map[Char, Int]
  ): Seq[Array[Int]] = {
    sentences.map { s =>
      sentenceToPaddedVec(s, maxLength, pad, vocabulary)
    }
  }
}
