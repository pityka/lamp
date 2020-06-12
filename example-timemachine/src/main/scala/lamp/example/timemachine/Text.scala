package lamp.example.timemachine

import lamp.data.Device
import cats.effect.Resource
import cats.effect.IO
import lamp.autograd.TensorHelpers
import org.saddle._
import aten.ATen
import aten.Tensor
import lamp.data.BatchStream

object Text {
  def englishToIntegers(text: String) = {
    val chars = text.toSeq
      .groupBy(identity)
      .toSeq
      .map {
        case (char, group) => (char, group.size)
      }
      .sortBy(_._2)
      .reverse
      .take(30)
      .map(_._1)
      .zipWithIndex
      .toMap
    val unknown = chars.size

    (chars, text.map(c => chars.get(c).getOrElse(unknown)).toVector)
  }
  def englishToIntegers(text: String, chars: Map[Char, Int]) = {

    val unknown = chars.size

    text.map(c => chars.get(c).getOrElse(unknown)).toVector
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
          TensorHelpers.fromLongVec(pairs.flatMap(_._1).toVec)
        val flattenedTarget =
          TensorHelpers.fromLongVec(pairs.flatMap(_._2).toVec)
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
        val movedFeature = device.to(transposedFeatures)
        val movedTarget = device.to(transposedTarget)
        Tensor.releaseAll(
          Array(
            viewedTarget,
            viewedFeature,
            flattenedTarget,
            flattenedFeature,
            transposedTarget,
            transposedFeatures
          )
        )

        Some((movedFeature, movedTarget)): Option[(Tensor, Tensor)]
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
