package lamp.data

import cats.effect.Resource
import cats.effect.IO
import lamp.autograd.{TensorHelpers, const}
import org.saddle._
import aten.ATen
import aten.Tensor
import lamp.nn.Module
import lamp.nn.StatefulModule
import lamp.syntax
import lamp.Device
import lamp.autograd.Variable
import lamp.autograd.ConcatenateAddNewDim
import lamp.FloatingPointPrecision
import lamp.DoublePrecision

object Text {
  def sequencePrediction[T](
      batch: Seq[Vector[Long]],
      device: Device,
      precision: FloatingPointPrecision,
      module: StatefulModule[T],
      init: T,
      vocabularySize: Int,
      steps: Int
  ): Resource[IO, Variable] = {
    val batchSize = batch.size
    def loop(
        lastOutput: Variable,
        lastState: T,
        n: Int,
        buffer: Seq[Variable]
    ): Seq[Variable] = {
      if (n == 0) buffer
      else {
        val (output, state) =
          module.forward1(lastOutput, lastState)

        val lastChar = if (output.shape(0) > 1) {
          val lastTimeStep1 =
            output.select(0, output.shape(0) - 1)

          lastTimeStep1.view((1L :: lastTimeStep1.shape).map(_.toInt))

        } else output

        val nextInput =
          lastChar.argmax(2, false).oneHot(vocabularySize).cast(precision)

        loop(
          nextInput,
          state,
          n - 1,
          buffer :+ nextInput
        )
      }
    }
    val batchAsOneHotEncodedTensor =
      makePredictionBatch(batch, device, vocabularySize, precision)
    batchAsOneHotEncodedTensor.flatMap { batch =>
      Resource.make(IO {

        ConcatenateAddNewDim(
          loop(const(batch), init, steps, Seq())
        ).value.view(List(steps, batchSize, vocabularySize))

      })(variable => IO { variable.releaseAll })
    }
  }

  /** Convert back to text. Tensor shape: time x batch x dim
    */
  def convertLogitsToText(
      tensor: Tensor,
      vocab: Map[Int, Char]
  ): Seq[String] = {

    val t = ATen.argmax(tensor, 2, false)
    val r = convertIntegersToText(t, vocab)
    t.release
    r

  }

  /** Convert back to text. Tensor shape: time x batch x dim
    */
  def convertIntegersToText(
      tensor: Tensor,
      vocab: Map[Int, Char]
  ): Seq[String] = {

    val r = TensorHelpers.toMatLong(tensor).T
    r.rows.map(v => v.toSeq.map(l => vocab(l.toInt)).mkString)

  }
  def charsToIntegers(text: String) = {
    val chars = text.toSeq
      .groupBy(identity)
      .toSeq
      .map {
        case (char, group) => (char, group.size)
      }
      .sortBy(_._2)
      .reverse
      .map(_._1)
      .zipWithIndex
      .toMap

    (chars, text.map(c => chars(c)).toVector)
  }
  def charsToIntegers(text: String, chars: Map[Char, Int]) = {

    text.map(c => chars(c)).toVector
  }

  def makePredictionBatch(
      examples: Seq[Vector[Long]],
      device: Device,
      vocabularSize: Int,
      precision: FloatingPointPrecision
  ) = {
    Resource.make(IO {

      val flattenedFeature =
        TensorHelpers.fromLongVec(examples.flatMap(identity).toVec)
      val viewedFeature = ATen._unsafe_view(
        flattenedFeature,
        Array(examples.size.toLong, examples.head.size.toLong)
      )
      val transposedFeatures = viewedFeature.transpose(0, 1)
      val movedFeature = device.to(transposedFeatures)
      val oneHot = ATen.one_hot(movedFeature, vocabularSize)
      val double =
        if (precision == DoublePrecision) ATen._cast_Double(oneHot, false)
        else ATen._cast_Float(oneHot, false)
      Tensor.releaseAll(
        Array(
          viewedFeature,
          flattenedFeature,
          transposedFeatures,
          movedFeature,
          oneHot
        )
      )

      double
    })(a => IO(a.release))
  }

  /**
    * Yields tensors of shape (time step x batch size)
    */
  def minibatchesFromText(
      text: Vector[Int],
      minibatchSize: Int,
      timeSteps: Int,
      device: Device
  ) = {
    def makeNonEmptyBatch(idx: Array[Int]) = {
      Resource.make(IO {
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
            .fromLongVec(pairs.flatMap(_._1).toVec, device)
        val flattenedTarget =
          TensorHelpers.fromLongVec(pairs.flatMap(_._2).toVec, device)
        val viewedFeature = ATen._unsafe_view(
          flattenedFeature,
          Array(idx.size.toLong, timeSteps.toLong)
        )
        val viewedTarget = ATen._unsafe_view(
          flattenedTarget,
          Array(idx.size.toLong, timeSteps.toLong)
        )
        val transposedFeatures =
          viewedFeature.transpose(0, 1)
        val transposedTarget = viewedTarget.transpose(0, 1)

        Tensor.releaseAll(
          Array(
            viewedTarget,
            viewedFeature,
            flattenedTarget,
            flattenedFeature
          )
        )

        Some((transposedFeatures, transposedTarget)): Option[(Tensor, Tensor)]
      }) {
        case None => IO.unit
        case Some((a, b)) =>
          IO {
            a.release
            b.release
          }
      }
    }
    val emptyResource = Resource.pure[IO, Option[(Tensor, Tensor)]](None)

    val dropped = text.drop(scala.util.Random.nextInt(timeSteps))
    val numSamples = (dropped.size - 1) / timeSteps
    val idx = array
      .shuffle(array.range(0, numSamples * timeSteps, timeSteps))
      .grouped(minibatchSize)
      .toList
      .dropRight(1)

    scribe.info(
      s"Total batches: ${idx.size}. Each $timeSteps token long and has $minibatchSize examples."
    )
    assert(idx.forall(_.size == minibatchSize))
    new BatchStream {
      private var remaining = idx
      def nextBatch: Resource[IO, Option[(Tensor, Tensor)]] =
        remaining match {
          case Nil => emptyResource
          case x :: tail =>
            val r = makeNonEmptyBatch(x)
            remaining = tail
            r
        }
    }

  }
}
