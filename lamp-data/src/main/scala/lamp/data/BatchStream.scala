package lamp.data

import cats.effect._
import org.saddle._
import lamp.autograd.{const}
import lamp.Device
import lamp.autograd.Variable
import lamp.Scope
import lamp.STen
import lamp.Movable
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicBoolean

sealed trait StreamControl[+I] {
  def map[B](f: I => B): StreamControl[B]
  def unsafeGet: I
}
object StreamControl {
  def apply[I](i: I): StreamControl[I] = NonEmptyBatch(i)
  implicit def StreamControlIsMovable[T: Movable] =
    new Movable[StreamControl[T]] {
      def list(m: StreamControl[T]) =
        m match {
          case EndStream            => Nil
          case EmptyBatch           => Nil
          case NonEmptyBatch(batch) => implicitly[Movable[T]].list(batch)
        }
    }
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

trait BatchStream[+I] { self =>

  /** May be called from multiple threads. */
  def nextBatch(device: Device): Resource[IO, StreamControl[(I, STen)]]

  def map[I2](f: (I, STen) => Resource[IO, StreamControl[(I2, STen)]]) =
    new BatchStream[I2] {
      def nextBatch(device: Device): Resource[IO, StreamControl[(I2, STen)]] =
        self
          .nextBatch(device)
          .flatMap(maybe =>
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

  def foldLeft[B](zero: B, device: Device)(
      f: (B, (I, STen)) => IO[B]
  ): IO[B] = {
    def loop(b: B): IO[B] = {
      nextBatch(device).allocated.flatMap {
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

  def single[A](resource: Resource[IO, StreamControl[(A, STen)]]) =
    new BatchStream[A] {
      private val pulled = new AtomicBoolean(false)
      def nextBatch(device: Device): Resource[IO, StreamControl[(A, STen)]] = {
        val old = pulled.getAndSet(true)
        if (!old)
          resource
        else Resource.pure[IO, StreamControl[(A, STen)]](EndStream)
      }

    }

  def fromIndices[A](
      indices: Array[Array[Int]]
  )(
      makeNonEmptyBatch: (
          Array[Int],
          Device
      ) => Resource[IO, StreamControl[(A, STen)]]
  ) =
    new BatchStream[A] {
      private val i = new AtomicInteger(0)
      private val N = indices.length
      def nextBatch(device: Device): Resource[IO, StreamControl[(A, STen)]] = {
        val old = i.getAndIncrement()
        if (old < N)
          makeNonEmptyBatch(indices(old), device)
        else Resource.pure[IO, StreamControl[(A, STen)]](EndStream)
      }

    }
  def fromIndicesAndBuckets[A, B](
      indices: Array[(Array[Int], Int)]
  )(
      loadBucket: Int => Resource[IO, B],
      makeNonEmptyBatch: (
          B,
          Array[Int],
          Device
      ) => Resource[IO, StreamControl[(A, STen)]]
  ) =
    new BatchStream[A] {
      private val i = new AtomicInteger(0)
      private val N = indices.length
      private var loadedBucketIndex = -1
      private var loadedBucket: Option[B] = None
      private var releaseBucket: Option[IO[Unit]] = None
      def nextBatch(device: Device): Resource[IO, StreamControl[(A, STen)]] = {
        val prev = i.getAndIncrement()
        if (prev < N) {
          val (indicesInBucket, bucketIndex) = indices(prev)
          val b =
            if (bucketIndex == loadedBucketIndex && loadedBucket.isDefined) {
              IO.pure((loadedBucket.get, IO.unit))
            } else
              for {
                _ <- releaseBucket.getOrElse(IO.unit)
                acq <- loadBucket(bucketIndex).allocated
                b = acq._1
                release = acq._2
                _ <- IO {
                  synchronized {
                    releaseBucket = Some(release)
                    loadedBucket = Some(b)
                    loadedBucketIndex = bucketIndex
                  }
                }
              } yield (b, IO.unit)
          Resource(b).flatMap { b =>
            makeNonEmptyBatch(b, indicesInBucket, device)
          }
        } else {
          val cleanup: Resource[IO, Unit] =
            if (releaseBucket.isEmpty) Resource.pure(())
            else Resource(IO.pure(((), releaseBucket.get)))
          cleanup.flatMap(_ =>
            Resource.pure[IO, StreamControl[(A, STen)]](EndStream)
          )
        }
      }

    }

  def scopeInResource =
    Resource.make(IO {
      Scope.free
    })(scope => IO { scope.release() })

  def minibatchesFromFull(
      minibatchSize: Int,
      dropLast: Boolean,
      features: STen,
      target: STen,
      rng: org.saddle.spire.random.Generator
  ) = {
    def makeNonEmptyBatch(idx: Array[Int], device: Device) = {
      scopeInResource.map { implicit scope =>
        val (d1, d2) = Scope { implicit scope =>
          val idxT = STen.fromLongVec(idx.toVec.map(_.toLong))
          val xcl = features.index(idxT)
          val tcl = target.index(idxT)
          val (d1, d2) = device.withOtherStreamThenSync(synchronizeBefore=false) {
            val d1 = device.to(xcl)
            val d2 = device.to(tcl)
            (d1, d2)
          }
          (d1, d2)
        }
        NonEmptyBatch((const(d1), d2)): StreamControl[(Variable, STen)]
      }

    }

    val idx = {
      val t = array
        .shuffle(array.range(0, features.sizes.head.toInt), rng)
        .grouped(minibatchSize)
        .toList
      if (dropLast) t.dropRight(1)
      else t
    }

    BatchStream.fromIndices(idx.toArray)(makeNonEmptyBatch)

  }

  def fromFullBatch(features: STen, targets: STen, device: Device) = {
    val resource = scopeInResource.map { implicit scope =>
      val xcl = device.to(features)
      val tcl = device.to(targets)
      NonEmptyBatch((const(xcl), tcl)): StreamControl[(Variable, STen)]

    }
    BatchStream.single(resource)
  }
}
