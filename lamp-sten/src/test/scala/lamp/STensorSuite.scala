package lamp

import org.saddle._
import org.saddle.ops.BinOps._

import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.compatible.Assertion
import aten.TensorOptions

class STenSuite extends AnyFunSuite {
  implicit def AssertionIsMovable = Movable.empty[Assertion]

  test("zeros cpu") {
    Scope.root { implicit scope =>
      val sum = Scope { implicit scope =>
        val ident = STen.eye(3, TensorOptions.d)
        val ones = STen.ones(List(3, 3), TensorOptions.d)
        ident + ones
      }
      assert(sum.toMat == mat.ones(3, 3) + mat.ident(3))
    }
  }
}
