package lamp.nn

import org.saddle._
import org.saddle.ops.BinOps._
import org.scalatest.funsuite.AnyFunSuite
import aten.ATen
import lamp.autograd._
import aten.TensorOptions
import org.scalatest.Tag

class AdamWSuite extends AnyFunSuite {

  def test1(id: String)(fun: Boolean => Unit) = {
    test(id) { fun(false) }
    test(id + "/CUDA", CudaTest) { fun(true) }
  }

  test1("AdamW noop") { cuda =>
    val initParams = mat.ones(1, 2)
    val params = TensorHelpers.fromMat(initParams, cuda)
    val gradients = TensorHelpers.fromMat(mat.zeros(1, 2), cuda)
    AdamW(
      parameters = List((params, NoTag)),
      learningRate = simple(1d),
      weightDecay = simple(0.00),
      beta1 = simple(1d),
      beta2 = simple(1d),
      scheduler = _ => 1d
    ).step(List(Some(gradients)))
    val updatedParams = TensorHelpers.toMat(params)
    assert(updatedParams == initParams)
  }
  test1("AdamW without weight decay") { cuda =>
    val initParams = mat.ones(1, 2)
    val params = TensorHelpers.fromMat(initParams, cuda)
    val gradients = TensorHelpers.fromMat(Mat(Vec(0.5, 0.75)).T, cuda)
    val opt = AdamW(
      parameters = List((params, NoTag)),
      learningRate = simple(0.1d),
      weightDecay = simple(0.00),
      beta1 = simple(0.999d),
      beta2 = simple(0.9d),
      scheduler = _ => 1d
    )
    opt.step(List(Some(gradients)))
    val updatedParams1 = TensorHelpers.toMat(params)
    assert(updatedParams1 == Mat(Vec(0.9999990000002, 0.9999990000001333)).T)
    opt.step(List(Some(gradients)))
    val updatedParams2 = TensorHelpers.toMat(params)
    assert(updatedParams2 == Mat(Vec(0.9999969728373086, 0.999996902616711)).T)
  }
  test1("AdamW") { cuda =>
    val initParams = mat.ones(1, 2)
    val params = TensorHelpers.fromMat(initParams, cuda)
    val gradients = TensorHelpers.fromMat(Mat(Vec(0.5, 0.75)).T, cuda)
    val opt = AdamW(
      parameters = List((params, NoTag)),
      learningRate = simple(0.1d),
      weightDecay = simple(0.00001),
      beta1 = simple(0.999d),
      beta2 = simple(0.9d),
      scheduler = _ => 1d
    )
    opt.step(List(Some(gradients)))
    val updatedParams1 = TensorHelpers.toMat(params)
    assert(
      updatedParams1.roundTo(10) == Mat(
        Vec(0.9999890000002, 0.9999890000001334)
      ).T.roundTo(10)
    )
    opt.step(List(Some(gradients)))
    val updatedParams2 = TensorHelpers.toMat(params)
    assert(
      updatedParams2.roundTo(10) == Mat(
        Vec(0.9999769729473086, 0.9999769027267111)
      ).T.roundTo(10)
    )
  }

}
