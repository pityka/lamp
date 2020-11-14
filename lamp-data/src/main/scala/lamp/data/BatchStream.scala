package lamp.data

import cats.effect._
import org.saddle._
import lamp.autograd.{const}
import lamp.Device
import lamp.autograd.Variable
import lamp.Scope
import lamp.STen

trait BatchStream[I] { self =>
  def nextBatch: Resource[IO, Option[(I, STen)]]

  def map[I2](f: (I, STen) => Resource[IO, (I2, STen)]) =
    new BatchStream[I2] {
      def nextBatch: Resource[IO, Option[(I2, STen)]] =
        self.nextBatch.flatMap(maybe =>
          maybe match {
            case None => Resource.pure[IO, Option[(I2, STen)]](None)
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
      features: STen,
      target: STen,
      device: Device,
      rng: org.saddle.spire.random.Generator
  ) = {
    def makeNonEmptyBatch(idx: Array[Int]) = {
      scopeInResource.map { implicit scope =>
        val (d1, d2) = Scope { implicit scope =>
          val idxT = STen.fromLongVec(idx.toVec.map(_.toLong))
          val xcl = features.index(idxT)
          val tcl = target.index(idxT)
          val d1 = device.to(xcl)
          val d2 = device.to(tcl)
          (d1, d2)
        }
        Some((const(d1), d2)): Option[(Variable, STen)]
      }

    }
    val emptyResource = Resource.pure[IO, Option[(Variable, STen)]](None)

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
      def nextBatch: Resource[IO, Option[(Variable, STen)]] =
        remaining match {
          case Nil => emptyResource
          case x :: tail =>
            val r = makeNonEmptyBatch(x)
            remaining = tail
            r
        }
    }

  }

  def fromFullBatch(features: STen, targets: STen, device: Device)(
      ) = {
    val resource = scopeInResource.map { implicit scope =>
      val xcl = device.to(features)
      val tcl = device.to(targets)
      Some((const(xcl), tcl)): Option[(Variable, STen)]

    }
    val emptyResource = Resource.pure[IO, Option[(Variable, STen)]](None)
    new BatchStream[Variable] {
      private var pulled = false
      def nextBatch: Resource[IO, Option[(Variable, STen)]] =
        if (!pulled) {
          pulled = true
          resource
        } else emptyResource
    }
  }
}
