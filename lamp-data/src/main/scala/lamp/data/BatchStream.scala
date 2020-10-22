package lamp.data

import cats.effect._
import aten.Tensor
import org.saddle._
import lamp.autograd.{const}
import lamp.TensorHelpers
import aten.ATen
import lamp.Device
import lamp.autograd.Variable
import lamp.Scope

trait BatchStream[I] { self =>
  def nextBatch: Resource[IO, Option[(I, Tensor)]]

  def map[I2](f: (I, Tensor) => Resource[IO, (I2, Tensor)]) =
    new BatchStream[I2] {
      def nextBatch: Resource[IO, Option[(I2, Tensor)]] =
        self.nextBatch.flatMap(maybe =>
          maybe match {
            case None => Resource.pure[IO, Option[(I2, Tensor)]](None)
            case Some(pair) =>
              f.tupled(pair).map(Some(_))
          }
        )
    }
}

object BatchStream {

  def scopeInResource =
    Resource.make(IO {
      Scope.free
    })(scope => IO { scope.release })

  def minibatchesFromFull(
      minibatchSize: Int,
      dropLast: Boolean,
      features: Tensor,
      target: Tensor,
      device: Device,
      rng: org.saddle.spire.random.Generator
  ) = {
    def makeNonEmptyBatch(idx: Array[Int]) = {
      scopeInResource.flatMap { implicit scope =>
        Resource.make(IO {
          val idxT = TensorHelpers.fromLongVec(idx.toVec.map(_.toLong))
          val xcl = ATen.index(features, Array(idxT))
          val tcl = ATen.index(target, Array(idxT))
          val d1 = device.to(xcl)
          val d2 = device.to(tcl)
          xcl.release
          tcl.release
          idxT.release
          Some((const(d1), d2)): Option[(Variable, Tensor)]
        }) {
          case None => IO.unit
          case Some((_, b)) =>
            IO {
              b.release
            }
        }
      }
    }
    val emptyResource = Resource.pure[IO, Option[(Variable, Tensor)]](None)

    val idx = {
      val t = array
        .shuffle(array.range(0, features.sizes.head.toInt), rng)
        .grouped(minibatchSize)
        .toList
      if (dropLast) t.dropRight(1)
      else t
    }

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

  def fromFullBatch(features: Tensor, targets: Tensor, device: Device)(
      ) = {
    val resource = scopeInResource.flatMap { implicit scope =>
      Resource.make(IO {
        val xcl = device.to(features)
        val tcl = device.to(targets)
        Some((const(xcl), tcl)): Option[(Variable, Tensor)]
      }) {
        case None => IO.unit
        case Some((_, b)) =>
          IO {
            b.release
          }
      }
    }
    val emptyResource = Resource.pure[IO, Option[(Variable, Tensor)]](None)
    new BatchStream[Variable] {
      private var pulled = false
      def nextBatch: Resource[IO, Option[(Variable, Tensor)]] =
        if (!pulled) {
          pulled = true
          resource
        } else emptyResource
    }
  }
}
