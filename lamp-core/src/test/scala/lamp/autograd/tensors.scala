package lamp.autograd

import org.saddle._
import org.scalatest.funsuite.AnyFunSuite
import aten.ATen
import aten.TensorOptions
import lamp.nn.CudaTest
import lamp.util.syntax
import lamp.Scope
import lamp.saddle.SaddleTensorHelpers

class SaddleTensorHelpersuite extends AnyFunSuite {

  test("to/from cuda", CudaTest) {
    val eye = ATen.eye_1(3, 3, TensorOptions.d.cuda)
    val m = SaddleTensorHelpers.toMat(eye)
    assert(m == mat.ident(3))
  }
  test("to/from double") {
    val eye = ATen.eye_1(3, 3, TensorOptions.d)
    val m = SaddleTensorHelpers.toMat(eye)
    assert(m == mat.ident(3))
  }
  test("to/from long") {
    val eye = ATen.eye_1(3, 3, TensorOptions.l)
    val m = SaddleTensorHelpers.toLongMat(eye)
    assert(m == mat.ident(3).map(_.toLong))
  }
  test("to/from float") {
    val eye = ATen.eye_1(3, 3, TensorOptions.f)
    val m = SaddleTensorHelpers.toFloatMat(eye)
    assert(m == mat.ident(3).map(_.toFloat))
  }
  test("to/from double scalar") {
    val eye = ATen.scalar_tensor(1d, TensorOptions.d)
    val m = SaddleTensorHelpers.toMat(eye)
    assert(m == mat.ones(1, 1))
  }
  test("to/from float scalar") {
    val eye = ATen.scalar_tensor(1d, TensorOptions.f)
    val m = SaddleTensorHelpers.toFloatMat(eye)
    assert(m == mat.ones(1, 1).map(_.toFloat))
  }
  test("to/from long scalar") {
    val eye = ATen.scalar_tensor(1d, TensorOptions.l)
    val m = SaddleTensorHelpers.toLongMat(eye)
    assert(m == mat.ones(1, 1).map(_.toLong))
  }
  test("to/from double Vec") {
    val eye = ATen.ones(Array(3L), TensorOptions.d)
    val m = SaddleTensorHelpers.toMat(eye)
    assert(m == mat.ones(1, 3))
  }
  test("to/from float Vec") {
    val eye = ATen.ones(Array(3L), TensorOptions.f)
    val m = SaddleTensorHelpers.toFloatMat(eye)
    assert(m == mat.ones(1, 3).map(_.toFloat))
  }
  test("to/from long Vec") {
    val eye = ATen.ones(Array(3L), TensorOptions.l)
    val m = SaddleTensorHelpers.toLongMat(eye)
    assert(m == mat.ones(1, 3).map(_.toLong))
  }

  test("index") {
    val m = SaddleTensorHelpers.fromMat(mat.ident(3))
    val idx = ATen.squeeze_0(SaddleTensorHelpers.fromLongMat(Mat(Vec(1L))))
    val m2 = ATen.index(m, Array(idx))
    assert(SaddleTensorHelpers.toMat(m2) == Mat(Vec(0d, 1d, 0d)).T)
  }
  test("fromMatList") {
    val m =
      SaddleTensorHelpers.fromMatList(Seq(mat.ident(2), mat.ident(2), mat.ident(2)))
    assert(m.shape == List(3, 2, 2))
    assert(
      m.toDoubleArray.toSeq == Seq(1d, 0d, 0d, 1d, 1d, 0d, 0d, 1d, 1d, 0d, 0d,
        1d)
    )
  }

  test("zero_") {
    val t = ATen.eye_0(3, TensorOptions.dtypeDouble())
    assert(SaddleTensorHelpers.toMat(t) == mat.ident(3))
    ATen.zero_(t)
    import org.saddle.ops.BinOps._
    assert(SaddleTensorHelpers.toMat(t) == mat.ident(3) * 0d)
  }
  test("one hot") {
    val t =
      SaddleTensorHelpers.fromLongMat(Mat(Vec(0L, 1L), Vec(1L, 1L), Vec(0L, 0L)).T)

    val t2 = ATen.one_hot(t, 4)
    assert(t2.shape == List(3, 2, 4))
  }
  test("transpose") {
    val mat2x3 = Mat(Vec(1d, 2d), Vec(3d, 4d), Vec(5d, 6d))
    val t2x3 = SaddleTensorHelpers.fromMat(mat2x3)
    val t3x2 = ATen.transpose(t2x3, 0, 1)
    assert(SaddleTensorHelpers.toMat(t3x2) == mat2x3.T)
  }
  ignore(" memory leak") {
    val m = mat.ones(3000, 3000)
    0 until 1000 foreach { _ =>
      Scope.root { implicit scope =>
        val t = lamp.saddle.fromMat(m)
        val t2 = lamp.saddle.fromMat(m)
        val t3 = lamp.saddle.fromMat(m)
        val t4 =
          new Concatenate(
            scope,
            List(
              const(t),
              const(t2),
              const(t3)
            ),
            1
          ).value
        assert(t4.shape == List(3000, 9000))
        ()
      }
    }
  }

}
