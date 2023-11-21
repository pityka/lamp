package lamp.data

import cats.effect._
import lamp.autograd.{const}
import lamp.Device
import lamp.autograd.Variable
import lamp.Scope
import lamp.STen
import lamp.Movable
import cats.effect.std.CountDownLatch
import scala.collection.compat.immutable.ArraySeq
import lamp.BufferPair

sealed trait StreamControl[+I] {
  def map[B](f: I => B): StreamControl[B]
  def unsafeGet: I
}
object StreamControl {
  def apply[I](i: I): StreamControl[I] = NonEmptyBatch(i)
  implicit def StreamControlIsMovable[T: Movable]: Movable[StreamControl[T]] =
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

/** A functional stateful stream of items
  *
  * lamp's training loops work from data presented in BatchStreams.
  *
  * An instance of BatchStream is an description of the data stream, it does not
  * by itself allocates or stores any data. The stream needs to be driven by an
  * interpreter. lamp.data.IOLoops and the companion object BatchStream contain
  * those interpreters to make something useful with a BatchStream.
  *
  * See the abstract members and the companion object for more documentation.
  *
  * @tparam I
  *   the item type , the stream will yield items of this type
  * @tparam S
  *   the state type, the stream will carry over and accumulate state of this
  *   type
  * @tparam C
  *   type of accessory resources (e.g. buffers), the stream might need an
  *   instance of this type for its working. The intended use for fixed,
  *   pre-allocated pinned buffer pairs to facilitate host-device copies. See
  *   lamp.Device.toBatched and lamp.BufferPair.
  */
trait BatchStream[+I, S, C] { self =>

  /** Initial value of the State */
  def init: S

  /** Allocation of a resource needed during the lifetime of the stream The
    * intended use is for transfer buffer pairs
    */
  def allocateBuffers(target: Device): Resource[IO, C]

  /** Returns the resource of the next item and the next state suspended in an
    * effect.
    *
    * Returned values are wrapped in StreamControl type which signals the
    * interpreter whether the stream is finished.
    *
    * May be called from different threads, but always in serial State should be
    * carried over in the state parameter and return type
    */
  def nextBatch(
      device: Device,
      buffers: C,
      state: S
  ): IO[(S, Resource[IO, StreamControl[I]])]

  /** Drives the stream and returns all the items from it in a scala collection.
    *
    * @param device
    * @return
    */
  def drainIntoSeq(
      device: Device
  ): Resource[IO, Vector[I]] = {

    def loop(
        batch: Resource[IO, StreamControl[I]],
        buffer: C,
        s0: S,
        acc: Vector[StreamControl[I]],
        releases: IO[Unit]
    ): IO[(Vector[StreamControl[I]], IO[Unit])] = {
      batch.allocated.flatMap { case (control, release) =>
        val acc1 = acc.appended(control)
        val releases1 = releases *> release

        if (control == EndStream) IO.pure(acc1 -> releases1)
        else
          this.nextBatch(device, buffer, s0).flatMap { case (s1, next) =>
            loop(next, buffer, s1, acc1, releases1)
          }
      }
    }
    this
      .allocateBuffers(device)
      .flatMap { buffers =>
        Resource.make(IO {
          this.nextBatch(device, buffers, this.init).flatMap {
            case (s1, next) =>
              loop(next, buffers, s1, Vector.empty[StreamControl[I]], IO.unit)
          }
        }.flatten) { case (_, r) => r }

      }
      .map { case (vector, _) =>
        vector.collect { case NonEmptyBatch(x) =>
          x
        }
      }
  }

  /** Returns a new stream with EmptyBatches filtered out
    */
  def withoutEmptyBatches =
    new BatchStream[I, S, C] {
      def init = self.init

      def allocateBuffers(target: Device): Resource[IO, C] =
        self.allocateBuffers(target)

      def nextBatch(
          device: Device,
          buffers: C,
          state: S
      ): IO[(S, Resource[IO, StreamControl[I]])] =
        self
          .nextBatch(device, buffers, state)
          .flatMap { case (state1, resource) =>
            val s = resource.map { maybe =>
              maybe match {
                case EmptyBatch =>
                  self.nextBatch(device, buffers, state1)
                case x =>
                  IO.pure(
                    (state1, Resource.pure[IO, StreamControl[I]](x))
                  )

              }

            }

            s.allocated.flatMap { case (next, release) =>
              next.map { case (state, resource) =>
                (state, resource.onFinalize(release))
              }
            }

          }
    }

  /** Returns a stream which is the concatenation of this stream and an other.
    */
  def concat[I2 >: I, S2](other: BatchStream[I2, S2, C]) =
    new BatchStream[I2, (Boolean, Either[S, S2]), C] {
      def init = (false, Left(self.init))

      def allocateBuffers(target: Device): Resource[IO, C] =
        self.allocateBuffers(target)

      def nextBatch(
          device: Device,
          buffers: C,
          state: (Boolean, Either[S, S2])
      ): IO[((Boolean, Either[S, S2]), Resource[IO, StreamControl[I2]])] = {
        val current =
          if (!state._1)
            self.nextBatch(device, buffers, state._2.left.toOption.get)
          else other.nextBatch(device, buffers, state._2.toOption.get)

        current
          .flatMap { case (state1, resource) =>
            val s = resource.map { maybe =>
              maybe match {
                case EndStream =>
                  if (state._1)
                    IO.pure(
                      (
                        (true, Right(state1.asInstanceOf[S2])),
                        Resource.pure[IO, StreamControl[I2]](EndStream)
                      )
                    )
                  else
                    other.nextBatch(device, buffers, other.init).map {
                      case (s2, resource) =>
                        ((true, Right(s2)), resource)
                    }
                case x =>
                  IO.pure(
                    (
                      (false, Left(state1.asInstanceOf[S])),
                      Resource.pure[IO, StreamControl[I2]](x)
                    )
                  )

              }

            }

            s.allocated.flatMap { case (next, release) =>
              next.map { case (state, resource) =>
                (state, resource.onFinalize(release))
              }
            }

          }
      }
    }

  /** Returns a stream wich contains only the first n elements of this stream */
  def take(n: Long) =
    new BatchStream[I, (Long, S), C] {
      def init = (0L, self.init)

      def allocateBuffers(target: Device): Resource[IO, C] =
        self.allocateBuffers(target)
      def nextBatch(
          device: Device,
          buffers: C,
          state: (Long, S)
      ): IO[((Long, S), Resource[IO, StreamControl[I]])] =
        if (state._1 < n)
          self
            .nextBatch(device, buffers, state._2)
            .map(s => (state._1 + 1L, s._1) -> s._2)
        else
          IO.pure(
            (
              state._1 + 1L -> state._2,
              Resource.pure[IO, StreamControl[I]](EndStream)
            )
          )
    }

  /** Maps f over the elements of this stream */
  def map[I2](f: I => Resource[IO, StreamControl[I2]]) =
    new BatchStream[I2, S, C] {
      def init = self.init

      def allocateBuffers(target: Device): Resource[IO, C] =
        self.allocateBuffers(target)
      def nextBatch(
          device: Device,
          buffers: C,
          state: S
      ): IO[(S, Resource[IO, StreamControl[I2]])] =
        self
          .nextBatch(device, buffers, state)
          .map { case (state1, resource) =>
            (
              state1,
              resource.flatMap(maybe =>
                maybe match {
                  case EndStream =>
                    Resource.pure[IO, StreamControl[I2]](EndStream)
                  case EmptyBatch =>
                    Resource.pure[IO, StreamControl[I2]](EmptyBatch)
                  case NonEmptyBatch(i) =>
                    f(i)
                }
              )
            )
          }
    }

  /** Folds a function from an initial value over this stream */
  def foldLeft[B](zero: B, device: Device, stateZero: S)(
      f: (B, I) => IO[B]
  ): IO[B] = {
    allocateBuffers(device).use { buffers =>
      def loop(b: B, init: S): IO[B] = {
        nextBatch(device, buffers, init).flatMap { case (state1, resource) =>
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

  /** ensures the length of the stream is fixed. Either repeats an element or
    * truncates
    */
  def repeatOrTake(requiredLength: Long) =
    new BatchStream[I, (Long, S), C] {
      val init0 = self.init
      def init = (0L, self.init)

      def allocateBuffers(target: Device) =
        self.allocateBuffers(target)

      def nextBatch(
          device: Device,
          buffers: C,
          state: (Long, S)
      ) =
        if (state._1 >= requiredLength)
          IO.pure(
            (state._1 + 1L, state._2) -> Resource.pure[IO, StreamControl[I]](
              EndStream
            )
          )
        else {
          self
            .nextBatch(device, buffers, state._2)
            .flatMap { case (state1, resource) =>
              val s = resource.map { maybe =>
                maybe match {
                  case EndStream =>
                    self.nextBatch(device, buffers, init0).map {
                      case (s2, resource) =>
                        ((state._1 + 1, s2), resource)
                    }
                  case x =>
                    IO.pure(
                      (
                        (state._1 + 1, state1),
                        Resource.pure[IO, StreamControl[I]](x)
                      )
                    )

                }

              }

              s.allocated.flatMap { case (next, release) =>
                next.map { case (state, resource) =>
                  (state, resource.onFinalize(release))
                }
              }

            }
        }

    }

  /** Takes only batches where (i % n == offset), i being the number of batch
    * counted from 0
    * @return
    */
  def everyNth(n: Int, offset: Int) =
    new BatchStream[I, (Long, S), C] {
      def init = (0L, self.init)

      def allocateBuffers(target: Device) =
        self.allocateBuffers(target)

      def nextBatch(
          device: Device,
          buffers: C,
          state: (Long, S)
      ) = {
        self
          .nextBatch(device, buffers, state._2)
          .flatMap { case (state1, resource) =>
            if (state._1 % n == offset)
              IO.pure(((state._1 + 1L, state1), resource))
            else nextBatch(device, buffers, (state._1 + 1L, state1))

          }
      }

    }

}

object BatchStream {

  /** Creates a stream from a single item */
  def single[A](resource: Resource[IO, StreamControl[A]]) =
    new BatchStream[A, Boolean, Unit] {
      def init = false

      def allocateBuffers(target: Device): Resource[IO, Unit] =
        Resource.unit[IO]

      def nextBatch(
          device: Device,
          buffers: Unit,
          pulled: Boolean
      ) = IO {
        if (!pulled)
          (true, resource)
        else (true, Resource.pure[IO, StreamControl[A]](EndStream))
      }

    }

  /** Creates a stream from a vector of items */
  def fromVector[A](resources: Vector[Resource[IO, StreamControl[A]]]) =
    new BatchStream[A, Int, Unit] {
      def init = 0

      def allocateBuffers(target: Device): Resource[IO, Unit] =
        Resource.unit[IO]

      def nextBatch(
          device: Device,
          buffers: Unit,
          i: Int
      ) = IO {
        if (i < resources.size)
          (i + 1, resources(i))
        else (i + 1, Resource.pure[IO, StreamControl[A]](EndStream))
      }

    }

  /** Creates a stream from an array of indices and a lambda using a subset of
    * those indexes to allocate the batch
    *
    * The indices refer to some other external data structure
    */
  def fromIndicesWithBuffers[A, C](
      indices: Array[Array[Int]],
      allocateBuffers1: Device => Resource[IO, C]
  )(
      makeNonEmptyBatch: (
          Array[Int],
          C,
          Device
      ) => Resource[IO, StreamControl[A]]
  ) =
    new BatchStream[A, Int, C] {
      def init = 0
      def allocateBuffers(target: Device): Resource[IO, C] = allocateBuffers1(
        target
      )
      private val N = indices.length
      def nextBatch(device: Device, buffers: C, counter: Int) = IO {
        if (counter < N)
          (counter + 1, makeNonEmptyBatch(indices(counter), buffers, device))
        else
          (counter + 1, Resource.pure[IO, StreamControl[A]](EndStream))
      }

    }
  def fromFunctionWithBuffers[A, C](
      numBatches: Int,
      allocateBuffers1: Device => Resource[IO, C]
  )(
      makeNonEmptyBatch: (
          C,
          Device
      ) => Resource[IO, StreamControl[A]]
  ) =
    new BatchStream[A, Int, C] {
      def init = 0
      def allocateBuffers(target: Device): Resource[IO, C] = allocateBuffers1(
        target
      )
      def nextBatch(device: Device, buffers: C, counter: Int) = IO {
        if (counter < numBatches)
          (counter + 1, makeNonEmptyBatch(buffers, device))
        else
          (counter + 1, Resource.pure[IO, StreamControl[A]](EndStream))
      }

    }

  /** Creates a stream from an array of indices and a lambda using a subset of
    * those indexes to allocate the batch
    *
    * The indices refer to some other external data structure
    */
  def fromIndices[A, C](
      indices: Array[Array[Int]]
  )(
      makeNonEmptyBatch: (
          Array[Int],
          Device
      ) => Resource[IO, StreamControl[(A, STen)]]
  ) = fromIndicesWithBuffers(indices, _ => Resource.unit)((a, _, c) =>
    makeNonEmptyBatch(a, c)
  )
  def fromFunction[A, C](
      numBatches: Int,
      makeNonEmptyBatch: (
          Device
      ) => Resource[IO, StreamControl[(A, STen)]]
  ) = fromFunctionWithBuffers(numBatches, _ => Resource.unit)((_, c) =>
    makeNonEmptyBatch(c)
  )

  private[lamp] def scopeInResource =
    Resource.make(IO {
      Scope.free
    })(scope => IO { scope.release() })

  /** Create a stream from the first dimension of a tensor */
  def minibatchesFromFull(
      minibatchSize: Int,
      dropLast: Boolean,
      features: STen,
      target: STen,
      rng: scala.util.Random
  ) = {
    def makeNonEmptyBatch(
        idx: Array[Int],
        buffers: BufferPair,
        device: Device
    ) = {
      scopeInResource.evalMap { implicit scope =>
        IO.interruptible {
          val (d1, d2) = Scope { implicit scope =>
            val idxT = STen.fromLongArray(
              idx.map(_.toLong),
              List(idx.length),
              features.device
            )
            val xcl = features.index(idxT)
            val tcl = target.index(idxT)
            val (d1, d2) =
              device.withOtherStream(
                synchronizeBefore = false,
                synchronizeAfter = true
              ) {

                val d1 = device.toBatched(List(xcl), buffers).head
                val d2 = device.to(tcl)
                (d1, d2)
              }
            (d1, d2)
          }
          NonEmptyBatch((const(d1), d2)): StreamControl[(Variable, STen)]
        }
      }

    }

    val allocateBuffers = (device: Device) =>
      Scope.inResource.map({ implicit scope =>
        val featureBufferSize =
          features.shape.drop(1).foldLeft(1L)(_ * _) * minibatchSize
        device.allocateBuffers(featureBufferSize, features.options)
      })

    val idx = {
      val t = rng
        .shuffle(
          ArraySeq.unsafeWrapArray(Array.range(0, features.sizes.head.toInt))
        )
        .grouped(minibatchSize)
        .toList
        .map(_.toArray)
      if (dropLast) t.dropRight(1)
      else t
    }

    BatchStream.fromIndicesWithBuffers[(Variable, STen), BufferPair](
      idx.toArray,
      allocateBuffers
    )(makeNonEmptyBatch)

  }

  /** Create a stream of a single full batch  of features and targets */
  def fromFullBatch(features: STen, targets: STen, device: Device) = {
    val resource = scopeInResource.map { implicit scope =>
      val xcl = device.to(features)
      val tcl = device.to(targets)
      NonEmptyBatch((const(xcl), tcl)): StreamControl[(Variable, STen)]

    }
    BatchStream.single(resource)
  }

  private[data] object StagedLoader {

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
      def nextBatch[C](
          batchIdxWithinBucket: Int,
          device: Device,
          buffers: C,
          loadBatch: (
              B,
              Array[Int],
              C,
              Device
          ) => Resource[IO, StreamControl[A]]
      ): IO[Resource[IO, StreamControl[A]]] = {

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
                      loadBatch(b, indices.toArray, buffers, device)
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
      def nextBatch[C](
          device: Device,
          buffers: C,
          loadBatch: (
              B,
              Array[Int],
              C,
              Device
          ) => Resource[IO, StreamControl[A]]
      ): IO[(State[A, B], Resource[IO, StreamControl[A]])] = {
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
                  s.asInstanceOf[NotYetOpen[A, B]].open().map(Some(_))

              }
            }

          openNextBucket.flatMap { nextBucketOpen =>
            bucketState match {
              case s @ NotYetOpen(_, _) =>
                s.asInstanceOf[NotYetOpen[A, B]]
                  .open()
                  .flatMap { opened =>
                    val nextState = copy(
                      batchIdx = batchIdx + 1,
                      buckets = updateBuckets(
                        buckets,
                        bucketIdx,
                        Some(opened), // .asInstanceOf[Opened[A,B]]),
                        nextBucketOpen // .asInstanceOf[Option[Opened[A, B]]]
                      )
                    )
                    opened
                      .nextBatch(
                        batchIdxWithinBucket,
                        device,
                        buffers,
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
                s.asInstanceOf[Opened[A, B]]
                  .nextBatch(
                    batchIdxWithinBucket,
                    device,
                    buffers,
                    loadBatch
                  )
                  .map((nextState, _))

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

  /** A two stage data loader which first loads items of type B, then from B
    * loads items of type A
    *
    * Makes sense if loading B is quicker than loading an equivalent amount of A
    * e.g. because B is a preformed batch of A-s on secondary medium
    */
  def stagedFromIndices[A, B, C](
      indices: Array[Array[Int]],
      bucketSize: Int,
      allocateBuffers0: Device => Resource[IO, C]
  )(
      loadInstancesToStaging: Array[Int] => Resource[IO, B],
      makeNonEmptyBatch: (
          B,
          Array[Int],
          C,
          Device
      ) => Resource[IO, StreamControl[A]]
  ) =
    new BatchStream[A, StagedLoader.State[A, B], C] {

      def allocateBuffers(target: Device) = allocateBuffers0(target)

      private val bucketIndices: Vector[StagedLoader.BucketIndices] =
        indices.grouped(bucketSize).toVector.map { originalIndices =>
          val instancesWithOriginalIndices =
            originalIndices.flatten.distinct.sorted

          StagedLoader.BucketIndices(
            instancesWithOriginalIndices = instancesWithOriginalIndices,
            bucketSpecificIndices = originalIndices.map { minibatch =>
              minibatch
                .map(i => instancesWithOriginalIndices.indexWhere(_ == i))

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
          buffers: C,
          state: StagedLoader.State[A, B]
      ): IO[
        (StagedLoader.State[A, B], Resource[IO, StreamControl[A]])
      ] = {
        state.nextBatch(device, buffers, makeNonEmptyBatch)
      }

    }
}
