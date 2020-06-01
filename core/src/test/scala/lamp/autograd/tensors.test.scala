package lamp.autograd

import org.saddle._
import org.scalatest.funsuite.AnyFunSuite
import aten.ATen
import aten.TensorOptions
import lamp.nn.CudaTest

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
}
