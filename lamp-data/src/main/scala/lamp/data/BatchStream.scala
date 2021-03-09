package lamp.data

import cats.effect._
import org.saddle._
import lamp.autograd.{const}
import lamp.Device
import lamp.autograd.Variable
import lamp.Scope
import lamp.STen

sealed trait StreamControl[+I] {
  def map[B](f: I => B): StreamControl[B]
  def unsafeGet: I
}
object StreamControl {
  def apply[I](i: I): StreamControl[I] = NonEmptyBatch(i)
}
case object EmptyBatch extends StreamControl[Nothing] {
  def map[B](f: Nothing => B): StreamControl[B] = this
  def unsafeGet = throw new RuntimeException("get on EmptyBatch")
}
case object EndStream extends StreamControl[Nothing] {
  def map[B](f: Nothing => B): StreamControl[B] = this
  def unsafeGet = throw new RuntimeException("get on EndStream")

}
case class NonEmptyBatch[I](batch: I) extends StreamControl[I] {
  def map[B](f: I => B): StreamControl[B] = NonEmptyBatch(f(batch))
  def unsafeGet = batch
}

trait BatchStream[I] { self =>
  def nextBatch: Resource[IO, StreamControl[(I, STen)]]

  def map[I2](f: (I, STen) => Resource[IO, StreamControl[(I2, STen)]]) =
    new BatchStream[I2] {
      def nextBatch: Resource[IO, StreamControl[(I2, STen)]] =
        self.nextBatch.flatMap(maybe =>
          maybe match {
            case EndStream =>
              Resource.pure[IO, StreamControl[(I2, STen)]](EndStream)
            case EmptyBatch =>
              Resource.pure[IO, StreamControl[(I2, STen)]](EmptyBatch)
            case NonEmptyBatch(pair) =>
              f.tupled(pair)
          }
        )
    }

  def foldLeft[B](zero: B)(f: (B, (I, STen)) => IO[B]): IO[B] = {
    def loop(b: B): IO[B] = {
      nextBatch.allocated.flatMap {
        case (EndStream, release)  => release *> IO.pure(b)
        case (EmptyBatch, release) => release *> loop(b)
        case (NonEmptyBatch(batch), release) =>
          f(b, batch).attempt.flatMap { v =>
            release.flatMap { _ =>
              v match {
                case Left(e)  => IO.raiseError(e)
                case Right(b) => loop(b)
              }
            }
          }
      }
    }

    loop(zero)

  }
}

object BatchStream {

  def scopeInResource =
    Resource.make(IO {
      Scope.free
    })(scope => IO { scope.release() })

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
        NonEmptyBatch((const(d1), d2)): StreamControl[(Variable, STen)]
      }

    }
    val endStream =
      Resource.pure[IO, StreamControl[(Variable, STen)]](EndStream)

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
      def nextBatch: Resource[IO, StreamControl[(Variable, STen)]] =
        remaining match {
          case Nil => endStream
          case x :: tail =>
            val r = makeNonEmptyBatch(x)
            remaining = tail
            r
        }
    }

  }

  def fromFullBatch(features: STen, targets: STen, device: Device) = {
    val resource = scopeInResource.map { implicit scope =>
      val xcl = device.to(features)
      val tcl = device.to(targets)
      NonEmptyBatch((const(xcl), tcl)): StreamControl[(Variable, STen)]

    }
    val endStream =
      Resource.pure[IO, StreamControl[(Variable, STen)]](EndStream)
    new BatchStream[Variable] {
      private var pulled = false
      def nextBatch: Resource[IO, StreamControl[(Variable, STen)]] =
        if (!pulled) {
          pulled = true
          resource
        } else endStream
    }
  }
}
