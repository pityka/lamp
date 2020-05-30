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
}
