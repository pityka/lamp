package lamp.nn

import org.saddle._
import org.saddle.ops.BinOps._
import org.scalatest.funsuite.AnyFunSuite
import aten.ATen
import lamp.autograd._
import aten.TensorOptions
import org.scalatest.Tag

class SGDSuite extends AnyFunSuite {

  def test1(id: String)(fun: Boolean => Unit) = {
    test(id) { fun(false) }
    test(id + "/CUDA", CudaTest) { fun(true) }
  }

  test1("SGD noop") { cuda =>
    val initParams = mat.ones(1, 2)
    val params = TensorHelpers.fromMat(initParams, cuda)
    val gradients = TensorHelpers.fromMat(mat.zeros(1, 2), cuda)
    SGDW(
      parameters = List((params, NoTag)),
      learningRate = simple(1d),
      weightDecay = simple(0.00),
      momentum = None,
      scheduler = _ => 1d
    ).step(List(Some(gradients)))
    val updatedParams = TensorHelpers.toMat(params)
    assert(updatedParams == initParams)
  }
  test1("SGD without momentum, without weight decay") { cuda =>
    val initParams = mat.ones(1, 2)
    val params = TensorHelpers.fromMat(initParams, cuda)
    val gradients = TensorHelpers.fromMat(mat.ones(1, 2) * 0.5, cuda)
    SGDW(
      parameters = List((params, NoTag)),
      learningRate = simple(1d),
      weightDecay = simple(0.00),
      momentum = None,
      scheduler = _ => 1d
    ).step(List(Some(gradients)))
    val updatedParams = TensorHelpers.toMat(params)
    assert(updatedParams == initParams * 0.5)
  }
  test1("SGD without momentum") { cuda =>
    val initParams = mat.ones(1, 2)
    val params = TensorHelpers.fromMat(initParams, cuda)
    val gradients = TensorHelpers.fromMat(mat.ones(1, 2) * 0.5, cuda)
    SGDW(
      parameters = List((params, NoTag)),
      learningRate = simple(1d),
      weightDecay = simple(0.1),
      momentum = None,
      scheduler = _ => 1d
    ).step(List(Some(gradients)))
    val updatedParams = TensorHelpers.toMat(params)
    assert(updatedParams == (initParams * 0.5 - initParams * 0.1))
  }
  test1("SGD") { cuda =>
    val initParams = mat.ones(1, 2)
    val params = TensorHelpers.fromMat(initParams, cuda)
    val gradients = TensorHelpers.fromMat(Mat(Vec(0.5, 0.75)).T, cuda)
    val optim = SGDW(
      parameters = List((params, NoTag)),
      learningRate = simple(1d),
      weightDecay = simple(0.1),
      momentum = None,
      scheduler = _ => 1d
    )
    optim.step(List(Some(gradients)))
    val updatedParams1 = TensorHelpers.toMat(params)
    assert(
      updatedParams1
        .roundTo(4) == (initParams * Mat(Vec(0.5, 0.25)).T - initParams * 0.1)
        .roundTo(4)
    )
    optim.step(List(Some(gradients)))
    val updatedParams2 = TensorHelpers.toMat(params)
    assert(updatedParams2.roundTo(4) == Mat(Vec(-0.1400, -0.6150)).T)
  }
}
