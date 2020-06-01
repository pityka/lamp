package lamp.data

import cats.effect._
import aten.Tensor

trait BatchStream {
  def nextBatch: Resource[IO, Option[(Tensor, Tensor)]]
}

object BatchStream {
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
