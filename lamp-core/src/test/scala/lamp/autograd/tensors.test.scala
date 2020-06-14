package lamp.autograd

import org.saddle._
import org.scalatest.funsuite.AnyFunSuite
import aten.ATen
import aten.TensorOptions
import lamp.nn.CudaTest
import lamp.syntax
import lamp.util.NDArray

class TensorHelperSuite extends AnyFunSuite {
  test("to/from cuda", CudaTest) {
    val eye = ATen.eye_1(3, 3, TensorOptions.d.cuda)
    val m = TensorHelpers.toMat(eye)
    assert(m == mat.ident(3))
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
  ignore("cat - memory leak") {
    0 until 1000 foreach { _ =>
      val t = TensorHelpers.fromMat(mat.ones(3000, 3000))
      val t2 = TensorHelpers.fromMat(mat.ones(3000, 3000))
      val t3 = TensorHelpers.fromMat(mat.ones(3000, 3000))
      println(t3.weakUseCount())
      println(t3.useCount())
      val t4 =
        ConcatenateAddNewDim(
          List(
            const(t).copy(leaf = false),
            const(t2).copy(leaf = false),
            const(t3).copy(leaf = false)
          )
        ).value
      assert(t4.shape == List(3, 3000, 3000))

      t4.releaseAll

    }
  }

}
