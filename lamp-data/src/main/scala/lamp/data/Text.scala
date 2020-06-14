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
        loop(
          output,
          state,
          n - 1,
          buffer :+ output
        )
      }
    }
    val batchAsOneHotEncodedTensor =
      makePredictionBatch(batch, device, vocabularySize, precision)
    batchAsOneHotEncodedTensor.flatMap { batch =>
      Resource.make(IO {
        val (output, state0) = module.forward1(const(batch), init)
        val outputTimesteps = output.shape(0)
        val lastTimeStep1 =
          output.select(0, outputTimesteps - 1)
        val lastTimeStep =
          lastTimeStep1.view((1L :: lastTimeStep1.shape).map(_.toInt))

        val v = ConcatenateAddNewDim(
          loop(lastTimeStep, state0, steps, Seq())
        ).value.view(List(steps, batchSize, vocabularySize))

        v

      })(variable => IO { variable.releaseAll })
    }
  }

  /** Convert back to text. Tensor shape: time x batch x dim
    */
  def convertToText(tensor: Tensor, vocab: Map[Int, Char]): Seq[String] = {

    val t = ATen.argmax(tensor, 2, false)
    val r = TensorHelpers.toMatLong(t).T
    r.rows.map(v =>
      v.toSeq.map(l => vocab.get(l.toInt).getOrElse('#')).mkString
    )

  }
  def charsToIntegers(text: String) = {
    val chars = text.toLowerCase.toSeq
      .filterNot(c => c == '\n' || c == '\r')
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
    val unknown = chars.size

    (chars, text.map(c => chars.get(c).getOrElse(unknown)).toVector)
  }
  def charsToIntegers(text: String, chars: Map[Char, Int]) = {

    val unknown = chars.size

    text.map(c => chars.get(c).getOrElse(unknown)).toVector
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
      val transposedFeatures = ATen.t(viewedFeature)
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
        val transposedFeatures = ATen.t(viewedFeature)
        val transposedTarget = ATen.t(viewedTarget)

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
