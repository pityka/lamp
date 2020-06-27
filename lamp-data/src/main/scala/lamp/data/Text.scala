package lamp.data

import cats.effect.Resource
import cats.effect.IO
import lamp.autograd.{TensorHelpers, const}
import org.saddle._
import aten.ATen
import aten.Tensor
import lamp.nn.Module
import lamp.syntax
import lamp.Device
import lamp.autograd.Variable
import lamp.autograd.ConcatenateAddNewDim
import lamp.FloatingPointPrecision
import lamp.DoublePrecision
import cats.effect.ContextShift
import lamp.nn.StatefulModule
import lamp.nn.GenericModule
import lamp.nn.InitState

object Text {
  def sequencePrediction[T, M <: StatefulModule[Variable, Variable, T]](
      batch: Seq[Vector[Long]],
      device: Device,
      precision: FloatingPointPrecision,
      module: M with StatefulModule[Variable, Variable, T],
      steps: Int
  )(implicit is: InitState[M, T]): Resource[IO, Variable] = {
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
          module.forward((lastOutput, lastState))

        val lastChar = if (output.shape(0) > 1) {
          val lastTimeStep1 =
            output.select(0, output.shape(0) - 1)

          lastTimeStep1.view((1L :: lastTimeStep1.shape).map(_.toInt))

        } else output

        val nextInput =
          lastChar.argmax(2, false)

        loop(
          nextInput,
          state,
          n - 1,
          buffer :+ nextInput
        )
      }
    }

    makePredictionBatch(batch, device, precision).flatMap { batch =>
      Resource.make(IO {

        ConcatenateAddNewDim(
          loop(const(batch), module.initState, steps, Seq())
        ).value.view(List(steps, batchSize))

      })(variable => IO { variable.releaseAll })
    }
  }
  def sequencePredictionBeam[T, M <: StatefulModule[Variable, Variable, T]](
      prefix: Vector[Long],
      device: Device,
      precision: FloatingPointPrecision,
      module: M with StatefulModule[Variable, Variable, T],
      steps: Int
  )(implicit is: InitState[M, T]): Resource[IO, Seq[(Variable, Double)]] = {
    val batchSize = 1
    val k = 3
    def loop(
        n: Int,
        buffers: Seq[(Seq[(Variable, T)], Double)]
    ): Seq[(Seq[Variable], Double)] = {
      if (n == 0) {
        buffers.map(b => (b._1.map(_._1), b._2))
      } else {
        val candidates = buffers.flatMap {
          case (sequence, logProb0) =>
            val (lastOutput, lastState) = sequence.last

            val (output, state) =
              module.forward((lastOutput, lastState))

            val lastChar = if (output.shape(0) > 1) {
              val lastTimeStep1 =
                output.select(0, output.shape(0) - 1)

              lastTimeStep1.view((1L :: lastTimeStep1.shape).map(_.toInt))

            } else output

            (0 until lastChar.shape(2).toInt).map { i =>
              val selected = lastChar.select(2L, i.toLong)
              val tmp = Tensor.scalarLong(i.toLong, selected.options.toLong)
              val index = ATen._unsafe_view(tmp, Array(1L, 1L))
              tmp.release
              val logProb = selected.toMat.raw(0)
              (
                sequence,
                selected.assign(const(index).releasable),
                logProb + logProb0,
                state,
                i
              )
            }

        }

        val (chosen, rejected) = candidates.sortBy(_._3).reverse.splitAt(k)
        rejected.foreach(_._2.value.release)
        val newBuffers = chosen.map {
          case (sequence, selected, logProb, state, i) =>
            (sequence :+ ((selected, state)), logProb)
        }

        loop(
          n - 1,
          newBuffers
        )
      }
    }

    makePredictionBatch(Vector(prefix), device, precision).flatMap { batch =>
      Resource.make(IO {

        loop(steps, Seq(Seq((const(batch), module.initState)) -> 0d))
          .sortBy(_._2)
          .map {
            case (seq, logProb) =>
              val catted = ConcatenateAddNewDim(
                seq.map(v => v.select(0, v.shape(0) - 1))
              ).value.view(List(seq.size))

              (catted, logProb)
          }

      })(variables =>
        IO {
          ConcatenateAddNewDim(
            variables.map(_._1)
          ).value.releaseAll
        }
      )
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

    val r = TensorHelpers.toLongMat(tensor).T
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

      Tensor.releaseAll(
        Array(
          viewedFeature,
          flattenedFeature,
          transposedFeatures
        )
      )

      movedFeature
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

        Some((const(transposedFeatures).releasable, transposedTarget)): Option[
          (Variable, Tensor)
        ]
      }) {
        case None => IO.unit
        case Some((_, b)) =>
          IO {
            b.release
          }
      }
    }
    val emptyResource = Resource.pure[IO, Option[(Variable, Tensor)]](None)

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
    new BatchStream[Variable] {
      private var remaining = idx
      def nextBatch: Resource[IO, Option[(Variable, Tensor)]] =
        remaining match {
          case Nil => emptyResource
          case x :: tail =>
            val r = makeNonEmptyBatch(x)
            remaining = tail
            r
        }
    }

  }

  /**
    * Yields tensors of shape (time step x batch size)
    * @param text pairs of (source,destination) units
    */
  def minibatchesForSeq2Seq(
      text: IndexedSeq[(Vector[Long], Vector[Long])],
      minibatchSize: Int,
      timeSteps: Int,
      pad: Int,
      device: Device
  ): BatchStream[(Variable, Variable)] = {
    def makeNonEmptyBatch(idx: Array[Int]) = {
      Resource.make {
        val io = IO {
          val pairs = idx.map { i =>
            val segmentSource =
              text(i)._1
                .take(timeSteps)
                .padTo(timeSteps, pad.toLong)
            val segmentDestination =
              text(i)._2
                .take(timeSteps)
                .padTo(timeSteps, pad.toLong)
            val segmentTarget =
              text(i)._2
                .drop(1)
                .take(timeSteps)
                .padTo(timeSteps, pad.toLong)
            assert(segmentSource.length == segmentTarget.length)
            assert(segmentSource.length == segmentTarget.length)
            assert(segmentSource.length == segmentDestination.length)
            assert(segmentSource.length == timeSteps)

            (
              segmentSource,
              segmentDestination,
              segmentTarget
            )

          }

          val flattenedSource =
            TensorHelpers
              .fromLongVec(pairs.flatMap(_._1).toVec, device)
          val viewedSource = ATen._unsafe_view(
            flattenedSource,
            Array(idx.size.toLong, timeSteps.toLong)
          )
          val transposedSource =
            viewedSource.transpose(0, 1)
          val flattenedDest =
            TensorHelpers
              .fromLongVec(pairs.flatMap(_._2).toVec, device)
          val viewedDest = ATen._unsafe_view(
            flattenedDest,
            Array(idx.size.toLong, timeSteps.toLong)
          )
          val transposedDest =
            viewedDest.transpose(0, 1)
          val flattenedTarget =
            TensorHelpers.fromLongVec(pairs.flatMap(_._3).toVec, device)
          val viewedTarget = ATen._unsafe_view(
            flattenedTarget,
            Array(idx.size.toLong, timeSteps.toLong)
          )
          val transposedTarget = viewedTarget.transpose(0, 1)

          Tensor.releaseAll(
            Array(
              viewedTarget,
              viewedSource,
              viewedDest,
              flattenedTarget,
              flattenedSource,
              flattenedDest
            )
          )

          Some(
            (
              (
                const(transposedSource).releasable,
                const(transposedDest).releasable
              ),
              transposedTarget
            )
          ): Option[
            ((Variable, Variable), Tensor)
          ]
        }
        io
      } {
        case None => IO.unit
        case Some((_, b)) =>
          IO {
            b.release
          }
      }
    }
    val emptyResource =
      Resource.pure[IO, Option[((Variable, Variable), Tensor)]](None)

    val idx = array
      .shuffle(array.range(0, text.size))
      .grouped(minibatchSize)
      .toList
      .dropRight(1)

    scribe.info(
      s"Total batches: ${idx.size}. Each $timeSteps token long and has $minibatchSize examples."
    )
    assert(idx.forall(_.size == minibatchSize))
    new BatchStream[(Variable, Variable)] {
      private var remaining = idx
      def nextBatch: Resource[IO, Option[((Variable, Variable), Tensor)]] =
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
