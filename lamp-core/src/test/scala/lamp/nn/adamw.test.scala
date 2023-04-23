package lamp.nn

import org.saddle._
import org.scalatest.funsuite.AnyFunSuite
import lamp.saddle._
import lamp.Scope
import org.scalatest.compatible.Assertion

class AdamWSuite extends AnyFunSuite {
  implicit def AssertionIsMovable: lamp.EmptyMovable[Assertion] =
    lamp.Movable.empty[Assertion]

  def test1(id: String)(fun: Boolean => Unit) = {
    test(id) { fun(false) }
    test(id + "/CUDA", CudaTest) { fun(true) }
  }

  test1("AdamW without weight decay") { cuda =>
    Scope.root { implicit scope =>
      val initParams = mat.ones(1, 2)
      val params = lamp.saddle.fromMat(initParams, cuda)
      val gradients = lamp.saddle.fromMat(Mat(Vec(0.5, 0.75)).T, cuda)
      val opt = AdamW(
        parameters = List((params, NoTag)),
        learningRate = simple(0.1d),
        weightDecay = simple(0.00),
        beta1 = simple(0.999d),
        beta2 = simple(0.9d)
      )
      opt.step(List(Some(gradients)), 1d)
      val updatedParams1 = params.toMat
      assert(
        updatedParams1 == Mat(Vec(0.9000000063245549, 0.90000000421637)).T
      )
      opt.step(List(Some(gradients)), 1d)
      val updatedParams2 = params.toMat

      assert(
        updatedParams2 == Mat(Vec(0.8000000109128679, 0.8000000072752449)).T
      )
    }
  }
  test1("RAdam without weight decay") { cuda =>
    Scope.root { implicit scope =>
      val initParams = mat.ones(1, 2)
      val params = lamp.saddle.fromMat(initParams, cuda)
      val gradients = lamp.saddle.fromMat(Mat(Vec(0.5, 0.75)).T, cuda)
      val opt = RAdam(
        parameters = List((params, NoTag)),
        learningRate = simple(0.1d),
        weightDecay = simple(0.00),
        beta1 = simple(0.999d),
        beta2 = simple(0.9d)
      )
      opt.step(List(Some(gradients)), 1d)
      val updatedParams1 = params.toMat
      assert(
        updatedParams1 == Mat(Vec(0.95, 0.925)).T
      )
      opt.step(List(Some(gradients)), 1d)
      val updatedParams2 = params.toMat

      assert(
        updatedParams2 == Mat(Vec(0.8999999999999992, 0.849999999999999)).T
      )
    }
  }
  test1("AdamW") { cuda =>
    Scope.root { implicit scope =>
      val initParams = mat.ones(1, 2)
      val params = lamp.saddle.fromMat(initParams, cuda)
      val gradients = lamp.saddle.fromMat(Mat(Vec(0.5, 0.75)).T, cuda)
      val opt = AdamW(
        parameters = List((params, NoTag)),
        learningRate = simple(0.1d),
        weightDecay = simple(0.00001),
        beta1 = simple(0.999d),
        beta2 = simple(0.9d)
      )
      opt.step(List(Some(gradients)), 1d)
      val updatedParams1 = params.toMat
      assert(
        updatedParams1.roundTo(10) == Mat(
          Vec(0.8996837785585381, 0.8996837764503532)
        ).T.roundTo(10)
      )
      opt.step(List(Some(gradients)), 1d)
      val updatedParams2 = params.toMat

      assert(
        updatedParams2.roundTo(10) == Mat(
          Vec(0.7994876035234454, 0.7994875998862823)
        ).T.roundTo(10)
      )
    }
  }
  test1("AdamW - half") { cuda =>
    Scope.root { implicit scope =>
      val initParams = mat.ones(1, 2)
      val params = lamp.saddle.fromMat(initParams, cuda).castToHalf
      val gradients =
        lamp.saddle.fromMat(Mat(Vec(0.5, 0.75)).T, cuda).castToHalf
      val opt = AdamW(
        parameters = List((params, NoTag)),
        learningRate = simple(0.1d),
        weightDecay = simple(0.00001),
        beta1 = simple(0.999d),
        beta2 = simple(0.9d),
        mixedPrecision = true
      )
      opt.step(List(Some(gradients)), 1d)
      val updatedParams1 = params.toMat

      assert(
        updatedParams1.roundTo(10) == Mat(
          Vec(0.89990234375, 0.89990234375)
        ).T.roundTo(10)
      )
      opt.step(List(Some(gradients)), 1d)
      val updatedParams2 = params.toMat
      assert(
        updatedParams2.roundTo(10) == Mat(
          Vec(0.79931640625, 0.79931640625)
        ).T.roundTo(10)
      )
    }
  }
  test1("Yogi") { cuda =>
    Scope.root { implicit scope =>
      val initParams = mat.ones(1, 2)
      val params = lamp.saddle.fromMat(initParams, cuda)
      val gradients = lamp.saddle.fromMat(Mat(Vec(0.5, 0.75)).T, cuda)
      val opt = Yogi(
        parameters = List((params, NoTag)),
        learningRate = simple(1d),
        weightDecay = simple(0.00001),
        beta1 = simple(0.999d),
        beta2 = simple(0.9d)
      )
      opt.step(List(Some(gradients)), 1d)
      val updatedParams1 = params.toMat
      assert(
        updatedParams1.roundTo(4) == Mat(
          Vec(0.0019860079840320344, 0.0013215579227696672)
        ).T.roundTo(4)
      )
      opt.step(List(Some(gradients)), 1d)
      val updatedParams2 = params.toMat
      assert(
        updatedParams2.roundTo(4) == Mat(
          Vec(-0.3050151568303592, -0.1358345992884816)
        ).T.roundTo(4)
      )
    }
  }
}
