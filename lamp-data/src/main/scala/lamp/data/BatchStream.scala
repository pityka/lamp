package lamp.data

import cats.effect._
import lamp.autograd.{const}
import lamp.Device
import lamp.autograd.Variable
import lamp.Scope
import lamp.STen
import lamp.Movable
import cats.effect.std.CountDownLatch

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

trait BatchStream[+I, S] { self =>

  def init: S

  /** May be called from different threads, but always in serial State should be
    * carried over in the state parameter and return type
    */
  def nextBatch(
      device: Device,
      state: S
  ): IO[(S, Resource[IO, StreamControl[(I, STen)]])]

  def map[I2](f: (I, STen) => Resource[IO, StreamControl[(I2, STen)]]) =
    new BatchStream[I2, S] {
      def init = self.init
      def nextBatch(
          device: Device,
          state: S
      ): IO[(S, Resource[IO, StreamControl[(I2, STen)]])] =
        self
          .nextBatch(device, state)
          .map { case (state1, resource) =>
            (
              state1,
              resource.flatMap(maybe =>
                maybe match {
                  case EndStream =>
                    Resource.pure[IO, StreamControl[(I2, STen)]](EndStream)
                  case EmptyBatch =>
                    Resource.pure[IO, StreamControl[(I2, STen)]](EmptyBatch)
                  case NonEmptyBatch(pair) =>
                    f.tupled(pair)
                }
              )
            )
          }
    }

  def foldLeft[B](zero: B, device: Device, stateZero: S)(
      f: (B, (I, STen)) => IO[B]
  ): IO[B] = {
    def loop(b: B, init: S): IO[B] = {
      nextBatch(device, init).flatMap { case (state1, resource) =>
        resource.allocated.flatMap {
          case (EndStream, release)  => release *> IO.pure(b)
          case (EmptyBatch, release) => release *> loop(b, state1)
          case (NonEmptyBatch(batch), release) =>
            f(b, batch).attempt.flatMap { v =>
              release.flatMap { _ =>
                v match {
                  case Left(e)  => IO.raiseError(e)
                  case Right(b) => loop(b, state1)
                }
              }
            }
        }
      }
    }

    loop(zero, stateZero)

  }
}

object BatchStream {

  def single[A](resource: Resource[IO, StreamControl[(A, STen)]]) =
    new BatchStream[A, Boolean] {
      def init = false

      def nextBatch(
          device: Device,
          pulled: Boolean
      ) = IO {
        if (!pulled)
          (true, resource)
        else (true, Resource.pure[IO, StreamControl[(A, STen)]](EndStream))
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
    new BatchStream[A, Int] {
      def init = 0
      private val N = indices.length
      def nextBatch(device: Device, counter: Int) = IO {
        if (counter < N)
          (counter + 1, makeNonEmptyBatch(indices(counter), device))
        else
          (counter + 1, Resource.pure[IO, StreamControl[(A, STen)]](EndStream))
      }

    }

  private[lamp] def scopeInResource =
    Resource.make(IO {
      Scope.free
    })(scope => IO { scope.release() })

  def minibatchesFromFull(
      minibatchSize: Int,
      dropLast: Boolean,
      features: STen,
      target: STen,
      rng: scala.util.Random
  ) = {
    def makeNonEmptyBatch(idx: Array[Int], device: Device) = {
      scopeInResource.map { implicit scope =>
        val (d1, d2) = Scope { implicit scope =>
          val idxT = STen.fromLongArray(idx.map(_.toLong), List(idx.length),features.device)
          val xcl = features.index(idxT)
          val tcl = target.index(idxT)
          val (d1, d2) =
            device.withOtherStreamThenSync(synchronizeBefore = false) {
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
      val t = rng.shuffle(Array.range(0, features.sizes.head.toInt))
        .grouped(minibatchSize)
        .toList.map(_.toArray)
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

  object StagedLoader {

    private def updateBuckets[A, B](
        buckets: Vector[BucketState[A, B]],
        idx: Int,
        open1: Option[Opened[A, B]],
        open2: Option[Opened[A, B]]
    ) = {
      val s1 =
        if (open1.isDefined) buckets.updated(idx, open1.get) else buckets
      val s2 =
        if (open2.isDefined) s1.updated(idx + 1, open2.get) else s1
      s2
    }

     sealed trait BucketState[+A, +B]
     case class NotYetOpen[A, B](
        indices: BucketIndices,
        fn: Array[Int] => Resource[IO, B]
    ) extends BucketState[A, B] {
      def open(): IO[Opened[A, B]] =
        for {
          d <- Deferred[IO, OpenedBucketState[B]]
          refCompleted <- Ref[IO].of(false)
          latch <- CountDownLatch[IO](indices.bucketSpecificIndices.length)

          _ <- fn(
            indices.instancesWithOriginalIndices.toArray
          ).allocated.flatMap { case (b, release) =>
            (latch.await >> (release *> refCompleted.set(true))).start *>
              d.complete(OpenedBucketState(indices, b))

          }.start
        } yield Opened(d, latch, refCompleted)
    }

     case class Opened[A, B](
        deferred: Deferred[IO, OpenedBucketState[B]],
        latch: CountDownLatch[IO],
        isClosed: Ref[IO, Boolean]
    ) extends BucketState[A, B] {
      def nextBatch(
          batchIdxWithinBucket: Int,
          device: Device,
          loadBatch: (
              B,
              Array[Int],
              Device
          ) => Resource[IO, StreamControl[(A, STen)]]
      ): IO[Resource[IO, StreamControl[(A, STen)]]] = {

        deferred.get.flatMap { openBucketState =>
          if (
            batchIdxWithinBucket >= openBucketState.indices.bucketSpecificIndices.length
          ) IO.pure(Resource.pure(EndStream))
          else {
            isClosed.get.flatMap { isClosed =>
              if (isClosed)
                IO.raiseError(
                  new RuntimeException(
                    "closed? " + batchIdxWithinBucket
                  )
                )
              else {
                IO {
                  val indices = openBucketState.indices.bucketSpecificIndices(
                    batchIdxWithinBucket
                  )
                  val b = openBucketState.b

                  Resource
                    .make(IO.unit)(_ => latch.release)
                    .flatMap { _ =>
                      loadBatch(b, indices.toArray, device)
                    }
                }
              }
            }
          }
        }
      }
    }

    private[lamp] case class OpenedBucketState[B](
        indices: BucketIndices,
        b: B
    )

    private[lamp] case class State[A, B](
        bucketSize: Int,
        buckets: Vector[BucketState[A, B]],
        batchIdx: Int
    ) {
      def batchIdxToBucketIdx(bIdx: Int): (Int, Int) = {
        val bucketIdx = bIdx / bucketSize
        val idxInBucket = bIdx % bucketSize
        (bucketIdx, idxInBucket)
      }
      def nextBatch(
          device: Device,
          loadBatch: (
              B,
              Array[Int],
              Device
          ) => Resource[IO, StreamControl[(A, STen)]]
      ): IO[(State[A, B], Resource[IO, StreamControl[(A, STen)]])] = {
        val (bucketIdx, batchIdxWithinBucket) = batchIdxToBucketIdx(batchIdx)
        if (bucketIdx >= buckets.size) IO.pure((this, Resource.pure(EndStream)))
        else {
          val bucketState = buckets(bucketIdx)

          val openNextBucket =
            if (bucketIdx == buckets.size - 1) IO.pure(None)
            else {
              val nextBucketIdx = bucketIdx + 1
              buckets(nextBucketIdx) match {
                case Opened(_, _, _) => IO.pure(None)
                case s @ NotYetOpen(_, _) =>
                  s.open().map(Some(_))

              }
            }

          openNextBucket.flatMap { nextBucketOpen =>
            bucketState match {
              case s @ NotYetOpen(_, _) =>
                s.open()
                  .flatMap { opened =>
                    val nextState = copy(
                      batchIdx = batchIdx + 1,
                      buckets = updateBuckets(
                        buckets,
                        bucketIdx,
                        Some(opened),
                        nextBucketOpen
                      )
                    )
                    opened
                      .nextBatch(
                        batchIdxWithinBucket,
                        device,
                        loadBatch
                      )
                      .map((nextState, _))
                  }
              case s @ Opened(_, _, _) =>
                val nextState = copy(
                  batchIdx = batchIdx + 1,
                  buckets =
                    updateBuckets(buckets, bucketIdx, None, nextBucketOpen)
                )
                s.nextBatch(
                  batchIdxWithinBucket,
                  device,
                  loadBatch
                ).map((nextState, _))

            }
          }
        }

      }
    }

   private[lamp] case class BucketIndices(
        instancesWithOriginalIndices: Array[Int],
        bucketSpecificIndices: Array[Array[Int]]
    )

    private[lamp] def init[A, B](
        bucketSize: Int,
        bucketIndices: Vector[BucketIndices],
        loadInstancesToStaging: Array[Int] => Resource[IO, B]
    ) =
      State[A, B](
        bucketSize = bucketSize,
        buckets = bucketIndices.toSeq.toVector
          .map(NotYetOpen[A, B](_, loadInstancesToStaging)),
        batchIdx = 0
      )

  }

  def stagedFromIndices[A, B](
      indices: Array[Array[Int]],
      bucketSize: Int
  )(
      loadInstancesToStaging: Array[Int] => Resource[IO, B],
      makeNonEmptyBatch: (
          B,
          Array[Int],
          Device
      ) => Resource[IO, StreamControl[(A, STen)]]
  ) =
    new BatchStream[A, StagedLoader.State[A, B]] {

      private val bucketIndices: Vector[StagedLoader.BucketIndices] =
        indices.grouped(bucketSize).toVector.map { originalIndices =>
          val instancesWithOriginalIndices =
            originalIndices.flatten.distinct.sorted

          StagedLoader.BucketIndices(
            instancesWithOriginalIndices = instancesWithOriginalIndices,
            bucketSpecificIndices = originalIndices.map { minibatch =>
              minibatch
                .map(i => instancesWithOriginalIndices.find(_ == i).getOrElse(-1))
                
            }
          )

        }

      override def init =
        StagedLoader.init[A, B](
          bucketSize,
          bucketIndices.toSeq.toVector,
          loadInstancesToStaging
        )

      override def nextBatch(
          device: Device,
          state: StagedLoader.State[A, B]
      ): IO[
        (StagedLoader.State[A, B], Resource[IO, StreamControl[(A, STen)]])
      ] = {
        state.nextBatch(device, makeNonEmptyBatch)
      }

    }
}
