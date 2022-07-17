package lamp.nn

import org.saddle._
import org.scalatest.funsuite.AnyFunSuite
import lamp.saddle._
import lamp.Scope
import lamp.STen
import org.scalatest.compatible.Assertion

class ShampooSuite extends AnyFunSuite {
  implicit val AssertionIsMovable : lamp.EmptyMovable[Assertion] = lamp.Movable.empty[Assertion]

  def test1(id: String)(fun: Boolean => Unit) = {
    test(id) { fun(false) }
    test(id + "/CUDA", CudaTest) { fun(true) }
  }

  test1("Shampoo without weight decay") { cuda =>
    Scope.root { implicit scope =>
      val initParams = mat.ones(1, 2)
      val params = lamp.saddle.fromMat(initParams, cuda)
      val gradients = lamp.saddle.fromMat(Mat(Vec(0.5, 0.75)).T, cuda)
      val opt = Shampoo(
        parameters = List((params, NoTag)),
        learningRate = simple(0.1d),
      )
      opt.step(List(Some(gradients)), 1d)
      val updatedParams1 = params.toMat
      assert(
        updatedParams1.roundTo(4) == Mat(Vec(0.9445, 0.2101)).T
      )
      opt.step(List(Some(gradients)), 1d)
      val updatedParams2 = params.toMat

      assert(
        updatedParams2.roundTo(4) == Mat(Vec(0.9053, -0.4542)).T
      )
    }
  }
  test1("Shampoo without weight decay - diagonal") { cuda =>
    Scope.root { implicit scope =>
      val initParams = mat.ones(1, 513)
      val params = lamp.saddle.fromMat(initParams, cuda)
      val gradients = STen.rand(List(1,513), params.options)
      val opt = Shampoo(
        parameters = List((params, NoTag)),
        learningRate = simple(0.1d),
      )
      opt.step(List(Some(gradients)), 1d)
      // val updatedParams1 = params.toMat
      // assert(
      //   updatedParams1.roundTo(4) == Mat(Vec(0.9445, 0.2101)).T
      // )
      opt.step(List(Some(gradients)), 1d)
      // val updatedParams2 = params.toMat

      // assert(
      //   updatedParams2.roundTo(4) == Mat(Vec(0.9053, -0.4542)).T
      // )
    }
  }
  
}
