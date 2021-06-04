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
import _root_.lamp.STenOptions
import _root_.lamp.Scope

class BatchStreamSuite extends AnyFunSuite {
  test("staged loader") {

    @volatile var bucketLoad = 0
    @volatile var bucketReleases = 0
    @volatile var batchLoads = 0
    @volatile var batchUses = 0
    @volatile var batchReleases = 0

    def simpleLoop[S](
        batchStream: BatchStream[Vec[Int], S],
        state0: S
    ): IO[Long] = {
      batchStream
        .nextBatch(CPU, state0)
        .flatMap { case (state1, resource) =>
          resource
            .use { batch =>
              IO {
                (
                  state1,
                  batch.map(_ => batchUses += 1)
                )
              }
            }

        }
        .flatMap {
          case (_, EndStream) => IO.pure(0)
          case (s1, EmptyBatch) =>
            simpleLoop(batchStream, s1)
          case (s1, NonEmptyBatch(_)) =>
            simpleLoop(
              batchStream,
              s1
            )
        }

    }

    val stream = BatchStream.stagedFromIndices[Vec[Int], Vec[Int]](
      indices = array.range(0, 22).grouped(3).toArray,
      bucketSize = 2
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

    simpleLoop(stream, stream.init).unsafeRunSync()

    assert(bucketLoad == 4)
    assert(bucketReleases == -4)
    assert(batchReleases == -8)
    assert(batchLoads == 8)
    assert(batchUses == 8)

  }
}
