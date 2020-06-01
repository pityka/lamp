package lamp.data

import cats.effect._
import aten.Tensor
import org.saddle._
import lamp.autograd.TensorHelpers
import aten.ATen

trait BatchStream {
  def nextBatch: Resource[IO, Option[(Tensor, Tensor)]]
}

object BatchStream {
  def minibatchesFromFull(
      minibatchSize: Int,
      dropLast: Boolean,
      features: Tensor,
      target: Tensor,
      device: Device
  ) = {
    def makeNonEmptyBatch(idx: Array[Int]) = {
      Resource.make(IO {
        val idxT = TensorHelpers.fromLongVec(idx.toVec.map(_.toLong))
        val xcl = ATen.index(features, Array(idxT))
        val tcl = ATen.index(target, Array(idxT))
        val d1 = device.to(xcl)
        xcl.release
        val d2 = device.to(tcl)
        tcl.release
        Some((d1, d2)): Option[(Tensor, Tensor)]
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

    val idx = {
      val t = array
        .shuffle(array.range(0, features.sizes.head.toInt))
        .grouped(minibatchSize)
        .toList
      if (dropLast) t.dropRight(1)
      else t
    }

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

  def fromFullBatch(features: Tensor, targets: Tensor, device: Device) = {
    val resource = Resource.make(IO {
      val xcl = device.to(features)
      val tcl = device.to(targets)
      Some((xcl, tcl)): Option[(Tensor, Tensor)]
    }) {
      case None => IO.unit
      case Some((a, b)) =>
        IO {
          a.release
          b.release
        }
    }
    val emptyResource = Resource.pure[IO, Option[(Tensor, Tensor)]](None)
    new BatchStream {
      private var pulled = false
      def nextBatch: Resource[IO, Option[(Tensor, Tensor)]] =
        if (!pulled) {
          pulled = true
          resource
        } else emptyResource
    }
  }
}
