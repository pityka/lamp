package lamp.data

import org.scalatest.funsuite.AnyFunSuite
import org.saddle._
import _root_.lamp.Device
import cats.effect.kernel.Resource
import cats.effect.IO
import _root_.lamp.STen
// import _root_.lamp.Scope
// import _root_.lamp.STenOptions
import _root_.lamp.CPU
import cats.effect.unsafe.implicits.global
import cats.effect.syntax.all._
import _root_.lamp.STenOptions
import _root_.lamp.Scope
import lamp.data.BatchStream.StagedLoader.NotYetOpen
import lamp.data.BatchStream.StagedLoader.Opened
import cats.effect.Ref

class BatchStreamSuite extends AnyFunSuite {

  test("everyNth") {
    val program = for {
      alloc1 <- Ref[IO].of(0)
      dealloc1 <- Ref[IO].of(0)
      alloc2 <- Ref[IO].of(0)
      dealloc2 <- Ref[IO].of(0)
      alloc3 <- Ref[IO].of(0)
      dealloc3 <- Ref[IO].of(0)
      use <- Ref[IO].of(List.empty[Int])
      stream = BatchStream.fromVector(
        Vector(
          Resource.make(alloc1.update(_ + 1).map(_ => { NonEmptyBatch(1) }))(
            _ => dealloc1.update(_ + 1)
          ),
          Resource.make(alloc2.update(_ + 1).map(_ => NonEmptyBatch(2)))(_ =>
            dealloc2.update(_ + 1)
          ),
          Resource.make(alloc3.update(_ + 1).map(_ => NonEmptyBatch(3)))(_ =>
            dealloc3.update(_ + 1)
          )
        )
      )
      concat = stream.everyNth(3, 1)
      _ <- concat.drainIntoSeq(lamp.CPU).use(x => use.set(x.toList))
      _ <- use.get.map(l => assert(l == List(2)))
      _ <- alloc1.get.map(x => assert(x == 0))
      _ <- dealloc1.get.map(x => assert(x == 0))
      _ <- alloc2.get.map(x => assert(x == 1))
      _ <- dealloc2.get.map(x => assert(x == 1))
      _ <- alloc3.get.map(x => assert(x == 0))
      _ <- dealloc3.get.map(x => assert(x == 0))
    } yield ()

    program.unsafeRunSync()

  }
  test("everyNth 2") {
    val program = for {
      alloc1 <- Ref[IO].of(0)
      dealloc1 <- Ref[IO].of(0)
      alloc2 <- Ref[IO].of(0)
      dealloc2 <- Ref[IO].of(0)
      alloc3 <- Ref[IO].of(0)
      dealloc3 <- Ref[IO].of(0)
      use <- Ref[IO].of(List.empty[Int])
      stream = BatchStream.fromVector(
        Vector(
          Resource.make(alloc1.update(_ + 1).map(_ => NonEmptyBatch(1)))(_ =>
            dealloc1.update(_ + 1)
          ),
          Resource.make(alloc2.update(_ + 1).map(_ => NonEmptyBatch(2)))(_ =>
            dealloc2.update(_ + 1)
          ),
          Resource.make(alloc3.update(_ + 1).map(_ => NonEmptyBatch(3)))(_ =>
            dealloc3.update(_ + 1)
          )
        )
      )
      concat = stream.everyNth(3, 0)
      _ <- concat.drainIntoSeq(lamp.CPU).use(x => use.set(x.toList))
      _ <- use.get.map(l => assert(l == List(1)))
      _ <- alloc1.get.map(x => assert(x == 1))
      _ <- dealloc1.get.map(x => assert(x == 1))
      _ <- alloc2.get.map(x => assert(x == 0))
      _ <- dealloc2.get.map(x => assert(x == 0))
      _ <- alloc3.get.map(x => assert(x == 0))
      _ <- dealloc3.get.map(x => assert(x == 0))
    } yield ()

    program.unsafeRunSync()

  }
  test("everyNth 3") {
    val program = for {
      alloc1 <- Ref[IO].of(0)
      dealloc1 <- Ref[IO].of(0)
      alloc2 <- Ref[IO].of(0)
      dealloc2 <- Ref[IO].of(0)
      alloc3 <- Ref[IO].of(0)
      dealloc3 <- Ref[IO].of(0)
      use <- Ref[IO].of(List.empty[Int])
      stream = BatchStream.fromVector(
        Vector(
          Resource.make(alloc1.update(_ + 1).map(_ => NonEmptyBatch(1)))(_ =>
            dealloc1.update(_ + 1)
          ),
          Resource.make(alloc2.update(_ + 1).map(_ => NonEmptyBatch(2)))(_ =>
            dealloc2.update(_ + 1)
          ),
          Resource.make(alloc3.update(_ + 1).map(_ => NonEmptyBatch(3)))(_ =>
            dealloc3.update(_ + 1)
          )
        )
      )
      concat = stream.everyNth(3, 2)
      _ <- concat.drainIntoSeq(lamp.CPU).use(x => use.set(x.toList))
      _ <- use.get.map(l => assert(l == List(3)))
      _ <- alloc1.get.map(x => assert(x == 0))
      _ <- dealloc1.get.map(x => assert(x == 0))
      _ <- alloc2.get.map(x => assert(x == 0))
      _ <- dealloc2.get.map(x => assert(x == 0))
      _ <- alloc3.get.map(x => assert(x == 1))
      _ <- dealloc3.get.map(x => assert(x == 1))
    } yield ()

    program.unsafeRunSync()

  }
  test("batch stream single, repeat, drain") {
    val program = for {
      alloc1 <- Ref[IO].of(0)
      dealloc1 <- Ref[IO].of(0)
      use <- Ref[IO].of(List.empty[Int])
      one = BatchStream.single(
        Resource.make(alloc1.update(_ + 1).map(_ => NonEmptyBatch(1)))(_ =>
          dealloc1.update(_ + 1)
        )
      )
      concat = one.concat(one)
      c = one.repeatOrTake(4)
      _ <- c.drainIntoSeq(lamp.CPU).use(x => use.set(x.toList))
      _ <- use.get.map(l => assert(l == List(1, 1, 1, 1)))
      _ <- alloc1.get.map(x => assert(x == 4))
      _ <- dealloc1.get.map(x => assert(x == 4))
    } yield ()

    program.unsafeRunSync()

  }
  test("batch stream single, repeat, drain 2") {
    val program = for {
      alloc1 <- Ref[IO].of(0)
      dealloc1 <- Ref[IO].of(0)
      use <- Ref[IO].of(List.empty[Int])
      one = BatchStream.single(
        Resource.make(alloc1.update(_ + 1).map(_ => NonEmptyBatch(1)))(_ =>
          dealloc1.update(_ + 1)
        )
      )
      concat = one.concat(one)
      c = one.repeatOrTake(2)
      _ <- c.drainIntoSeq(lamp.CPU).use(x => use.set(x.toList))
      _ <- use.get.map(l => assert(l == List(1, 1)))
      _ <- alloc1.get.map(x => assert(x == 2))
      _ <- dealloc1.get.map(x => assert(x == 2))
    } yield ()

    program.unsafeRunSync()

  }
  test("batch stream single, repeat, drain 1") {
    val program = for {
      alloc1 <- Ref[IO].of(0)
      dealloc1 <- Ref[IO].of(0)
      use <- Ref[IO].of(List.empty[Int])
      one = BatchStream.single(
        Resource.make(alloc1.update(_ + 1).map(_ => NonEmptyBatch(1)))(_ =>
          dealloc1.update(_ + 1)
        )
      )
      concat = one.concat(one)
      c = one.repeatOrTake(1)
      _ <- c.drainIntoSeq(lamp.CPU).use(x => use.set(x.toList))
      _ <- use.get.map(l => assert(l == List(1)))
      _ <- alloc1.get.map(x => assert(x == 1))
      _ <- dealloc1.get.map(x => assert(x == 1))
    } yield ()

    program.unsafeRunSync()

  }
  test("batch stream single, concat, drain") {
    val program = for {
      alloc1 <- Ref[IO].of(false)
      dealloc1 <- Ref[IO].of(false)
      alloc2 <- Ref[IO].of(false)
      dealloc2 <- Ref[IO].of(false)
      use <- Ref[IO].of(List.empty[Int])
      one = BatchStream.single(
        Resource.make(alloc1.set(true).map(_ => NonEmptyBatch(1)))(_ =>
          dealloc1.set(true)
        )
      )
      two = BatchStream.single(
        Resource.make(alloc2.set(true).map(_ => NonEmptyBatch(2)))(_ =>
          dealloc2.set(true)
        )
      )
      c = one concat two
      _ <- c.drainIntoSeq(lamp.CPU).use(x => use.set(x.toList))
      _ <- use.get.map(l => assert(l == List(1, 2)))
      _ <- alloc1.get.map(x => assert(x))
      _ <- alloc2.get.map(x => assert(x))
      _ <- dealloc1.get.map(x => assert(x))
      _ <- dealloc2.get.map(x => assert(x))
    } yield ()

    program.unsafeRunSync()

  }

  test("staged loader") {

    @volatile var bucketLoad = 0
    @volatile var bucketReleases = 0
    @volatile var batchLoads = 0
    @volatile var batchUses = 0
    @volatile var batchReleases = 0
    @volatile var total = 0

    def simpleLoop[S](
        batchStream: BatchStream[(Vec[Int], STen), S, Unit],
        state0: S
    ): IO[S] = {
      batchStream
        .nextBatch(CPU, (), state0)
        .flatMap { case (state1, resource) =>
          resource
            .use { batch =>
              IO {
                (
                  state1,
                  batch.map { v =>
                    batchUses += 1
                    total += v._1.length
                  }
                )
              }
            }

        }
        .flatMap {
          case (s1, EndStream) =>
            IO.pure(s1)
          case (s1, EmptyBatch) =>
            simpleLoop(batchStream, s1)
          case (s1, NonEmptyBatch(_)) =>
            simpleLoop(
              batchStream,
              s1
            )
        }

    }

    val stream =
      BatchStream.stagedFromIndices[(Vec[Int], STen), Vec[Int], Unit](
        indices = array.range(0, 23).grouped(3).toArray,
        bucketSize = 3,
        _ => Resource.unit
      )(
        loadInstancesToStaging = (ar: Array[Int]) =>
          Resource.make(IO {
            bucketLoad += 1
            ar.toVec
          })(_ =>
            IO {
              bucketReleases -= 1
            }
          ),
        makeNonEmptyBatch = (
            bucket: Vec[Int],
            take: Array[Int],
            _: Unit,
            _: Device
        ) =>
          Resource.make(IO {
            batchLoads += 1
            NonEmptyBatch(
              (
                bucket.take(take),
                STen.zeros(List(1), STenOptions.d)(Scope.free)
              )
            )
          })(_ =>
            IO {
              batchReleases -= 1
            }
          )
      )

    val lastState = simpleLoop(stream, stream.init).unsafeRunSync()
    import scala.concurrent.duration._

    lastState.buckets
      .parTraverseN(1) {
        case NotYetOpen(_, _) =>
          IO.raiseError(new RuntimeException("Expected open or close"))
        case Opened(_, latch, _) => latch.await
      }
      .unsafeRunTimed(5 seconds)
    Thread.sleep(500)

    assert(bucketLoad == 3)
    assert(bucketReleases == -3)
    assert(batchReleases == -8)
    assert(batchLoads == 8)
    assert(batchUses == 8)
    assert(total == 23)

  }
}
