package lamp.data

import org.scalatest.funsuite.AnyFunSuite

import cats.effect.kernel.Resource
import cats.effect.IO
import _root_.lamp.STen

import cats.effect.unsafe.implicits.global
import cats.effect.syntax.all._

import _root_.lamp.Scope

class ScopeSuite extends AnyFunSuite {
  test("parallel alloc") {
    val stop = TensorLogger.start()(println _, (_, _) => true, 5000, 10000, 0)
    Scope.inResource
      .flatMap { implicit scope =>
        (0 until 1000).toList.parTraverseN(8)(_ =>
          Resource.make(IO {
            STen.zeros(List(30, 30))
          })(_ => IO.unit)
        )
      }
      .use { tensors =>
        IO {
          assert(tensors.size == 1000)
        }
      }
      .unsafeRunSync()

    stop.stop()
    TensorLogger.detailAllTensorOptions(println)
    assert(TensorLogger.queryActiveTensors().size == 0)

  }
}
