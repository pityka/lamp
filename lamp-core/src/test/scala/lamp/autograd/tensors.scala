package lamp.autograd

import org.saddle._
import org.scalatest.funsuite.AnyFunSuite
import aten.ATen
import aten.TensorOptions
import lamp.nn.CudaTest
import lamp.syntax
import lamp.util.NDArray

class TensorHelperSuite extends AnyFunSuite {
  implicit val pool = new AllocatedVariablePool
  test("to/from cuda", CudaTest) {
    val eye = ATen.eye_1(3, 3, TensorOptions.d.cuda)
    val m = TensorHelpers.toMat(eye)
    assert(m == mat.ident(3))
  }
  test("to/from double") {
    val eye = ATen.eye_1(3, 3, TensorOptions.d)
    val m = TensorHelpers.toMat(eye)
    assert(m == mat.ident(3))
  }
  test("to/from long") {
    val eye = ATen.eye_1(3, 3, TensorOptions.l)
    val m = TensorHelpers.toLongMat(eye)
    assert(m == mat.ident(3).map(_.toLong))
  }
  test("to/from float") {
    val eye = ATen.eye_1(3, 3, TensorOptions.f)
    val m = TensorHelpers.toFloatMat(eye)
    assert(m == mat.ident(3).map(_.toFloat))
  }
  test("to/from double scalar") {
    val eye = ATen.scalar_tensor(1d, TensorOptions.d)
    val m = TensorHelpers.toMat(eye)
    assert(m == mat.ones(1, 1))
  }
  test("to/from float scalar") {
    val eye = ATen.scalar_tensor(1d, TensorOptions.f)
    val m = TensorHelpers.toFloatMat(eye)
    assert(m == mat.ones(1, 1).map(_.toFloat))
  }
  test("to/from long scalar") {
    val eye = ATen.scalar_tensor(1d, TensorOptions.l)
    val m = TensorHelpers.toLongMat(eye)
    assert(m == mat.ones(1, 1).map(_.toLong))
  }
  test("to/from double Vec") {
    val eye = ATen.ones(Array(3L), TensorOptions.d)
    val m = TensorHelpers.toMat(eye)
    assert(m == mat.ones(1, 3))
  }
  test("to/from float Vec") {
    val eye = ATen.ones(Array(3L), TensorOptions.f)
    val m = TensorHelpers.toFloatMat(eye)
    assert(m == mat.ones(1, 3).map(_.toFloat))
  }
  test("to/from long Vec") {
    val eye = ATen.ones(Array(3L), TensorOptions.l)
    val m = TensorHelpers.toLongMat(eye)
    assert(m == mat.ones(1, 3).map(_.toLong))
  }

  test("index") {
    val m = TensorHelpers.fromMat(mat.ident(3))
    val idx = ATen.squeeze_0(TensorHelpers.fromLongMat(Mat(Vec(1L))))
    val m2 = ATen.index(m, Array(idx))
    assert(TensorHelpers.toMat(m2) == Mat(Vec(0d, 1d, 0d)).T)
  }
  test("fromMatList") {
    val m =
      TensorHelpers.fromMatList(Seq(mat.ident(2), mat.ident(2), mat.ident(2)))
    assert(m.shape == List(3, 2, 2))
    assert(
      m.toDoubleArray.toSeq == Seq(1d, 0d, 0d, 1d, 1d, 0d, 0d, 1d, 1d, 0d, 0d,
        1d)
    )
  }
  test("normalized") {
    val nd2x3x3 =
      NDArray((0 until 18).toArray.map(_.toDouble), List(2, 3, 3))
    val t = NDArray.tensorFromNDArray(nd2x3x3)
    val t2 = t.normalized.allocated.unsafeRunSync()._1
    val n2 = NDArray.tensorToNDArray(t2)
    assert(
      n2.toVec.roundTo(4) == Vec(-1d, -1d, -1d, -1d, -1d, -1d, -1d, -1d, -1d,
        1d, 1d, 1d, 1d, 1d, 1d, 1d, 1d, 1d)
    )

  }
  test("zero_") {
    val t = ATen.eye_0(3, TensorOptions.dtypeDouble())
    assert(t.toMat == mat.ident(3))
    ATen.zero_(t)
    import org.saddle.ops.BinOps._
    assert(t.toMat == mat.ident(3) * 0d)
  }
  test("one hot") {
    val t =
      TensorHelpers.fromLongMat(Mat(Vec(0L, 1L), Vec(1L, 1L), Vec(0L, 0L)).T)

    val t2 = ATen.one_hot(t, 4)
    assert(t2.shape == List(3, 2, 4))
  }
  test("transpose") {
    val mat2x3 = Mat(Vec(1d, 2d), Vec(3d, 4d), Vec(5d, 6d))
    val t2x3 = TensorHelpers.fromMat(mat2x3)
    val t3x2 = ATen.transpose(t2x3, 0, 1)
    assert(t3x2.toMat == mat2x3.T)
  }
  ignore(" memory leak") {
    val m = mat.ones(3000, 3000)
    0 until 1000 foreach { _ =>
      val t = TensorHelpers.fromMat(m)
      val t2 = TensorHelpers.fromMat(m)
      val t3 = TensorHelpers.fromMat(m)
      val t4 =
        Concatenate(
          List(
            const(t).releasable,
            const(t2).releasable,
            const(t3).releasable
          ),
          1
        ).value
      assert(t4.shape == List(3000, 9000))

      t4.releaseAll

    }
  }

}
