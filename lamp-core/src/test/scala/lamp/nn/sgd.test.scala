package lamp.nn

import org.saddle._
import org.saddle.ops.BinOps._
import org.scalatest.funsuite.AnyFunSuite
import lamp.Scope
import lamp.saddle._
import org.scalatest.compatible.Assertion

class SGDSuite extends AnyFunSuite {
implicit val AssertionIsMovable : lamp.EmptyMovable[Assertion] = lamp.Movable.empty[Assertion]
  def test1(id: String)(fun: Boolean => Unit) = {
    test(id) { fun(false) }
    test(id + "/CUDA", CudaTest) { fun(true) }
  }

  test1("SGD noop") { cuda =>
    Scope.root { implicit scope =>
      val initParams = mat.ones(1, 2)
      val params = lamp.saddle.fromMat(initParams, cuda)
      val gradients = lamp.saddle.fromMat(mat.zeros(1, 2), cuda)
      SGDW(
        parameters = List((params, NoTag)),
        learningRate = simple(1d),
        weightDecay = simple(0.00),
        momentum = None
      ).step(List(Some(gradients)), 1d)
      val updatedParams = params.toMat
      assert(updatedParams == initParams)
    }
  }
  test1("SGD without momentum, without weight decay") { cuda =>
    Scope.root { implicit scope =>
      val initParams = mat.ones(1, 2)
      val params = lamp.saddle.fromMat(initParams, cuda)
      val gradients = lamp.saddle.fromMat(mat.ones(1, 2) * 0.5, cuda)
      SGDW(
        parameters = List((params, NoTag)),
        learningRate = simple(1d),
        weightDecay = simple(0.00),
        momentum = None
      ).step(List(Some(gradients)), 1d)
      val updatedParams = params.toMat
      assert(updatedParams == initParams * 0.5)
    }
  }
  test1("SGD without momentum") { cuda =>
    Scope.root { implicit scope =>
      val initParams = mat.ones(1, 2)
      val params = lamp.saddle.fromMat(initParams, cuda)
      val gradients = lamp.saddle.fromMat(mat.ones(1, 2) * 0.5, cuda)
      SGDW(
        parameters = List((params, NoTag)),
        learningRate = simple(1d),
        weightDecay = simple(0.1),
        momentum = None
      ).step(List(Some(gradients)), 1d)
      val updatedParams = params.toMat
      assert(updatedParams == (initParams * 0.5 - initParams * 0.1))
    }
  }
  test1("SGD") { cuda =>
    Scope.root { implicit scope =>
      val initParams = mat.ones(1, 2)
      val params = lamp.saddle.fromMat(initParams, cuda)
      val gradients = lamp.saddle.fromMat(Mat(Vec(0.5, 0.75)).T, cuda)
      val optim = SGDW(
        parameters = List((params, NoTag)),
        learningRate = simple(1d),
        weightDecay = simple(0.1),
        momentum = None
      )
      optim.step(List(Some(gradients)), 1d)
      val updatedParams1 = params.toMat
      assert(
        updatedParams1
          .roundTo(4) == (initParams * Mat(Vec(0.5, 0.25)).T - initParams * 0.1)
          .roundTo(4)
      )
      optim.step(List(Some(gradients)), 1d)
      val updatedParams2 = params.toMat
      assert(updatedParams2.roundTo(4) == Mat(Vec(-0.1400, -0.6150)).T)

    }
  }
}
