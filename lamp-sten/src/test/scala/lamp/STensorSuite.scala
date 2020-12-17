package lamp

import org.saddle._
import org.saddle.linalg._
import org.saddle.ops.BinOps._

import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.compatible.Assertion

class STenSuite extends AnyFunSuite {
  implicit def AssertionIsMovable = Movable.empty[Assertion]

  test("zeros cpu") {
    Scope.root { implicit scope =>
      val sum = Scope { implicit scope =>
        val ident = STen.eye(3, STenOptions.d)
        val ones = STen.ones(List(3, 3), STenOptions.d)
        ident + ones
      }
      assert(sum.toMat == mat.ones(3, 3) + mat.ident(3))
    }
  }

  test("t") {
    Scope.root { implicit scope =>
      val t1 = STen.ones(List(3, 2), STenOptions.d)
      assert(t1.t.shape == List(2, 3))
    }
  }
  test("transpose") {
    Scope.root { implicit scope =>
      val t1 = STen.ones(List(3, 2, 4), STenOptions.d)
      assert(t1.transpose(1, 2).shape == List(3, 4, 2))
    }
  }
  test("select") {
    Scope.root { implicit scope =>
      val t1 = STen.ones(List(3, 2, 4), STenOptions.d)
      assert(t1.select(1, 1).shape == List(3, 4))
    }
  }
  test("indexSelect") {
    Scope.root { implicit scope =>
      val t1 = STen.ones(List(3, 2, 4), STenOptions.d)
      assert(
        t1.indexSelect(0, STen.fromLongVec(Vec(0L, 1L), false)).shape == List(
          2,
          2,
          4
        )
      )
    }
  }
  test("argmax") {
    Scope.root { implicit scope =>
      val t1 = STen.eye(3, STenOptions.d)
      assert(
        t1.argmax(0, keepDim = false).toLongVec == Vec(0, 1, 2)
      )
    }
  }
  test("argmin") {
    Scope.root { implicit scope =>
      val t1 = STen.eye(3, STenOptions.d).neg
      assert(
        t1.argmin(0, keepDim = false).toLongVec == Vec(0, 1, 2)
      )
    }
  }
  test("maskFill") {
    Scope.root { implicit scope =>
      val t1 = STen.eye(3, STenOptions.d)
      val mask = t1.equ(0d)
      assert(
        t1.maskFill(mask, -1).toMat.row(0) == Vec(1d, -1d, -1d)
      )
    }
  }
  test("cat") {
    Scope.root { implicit scope =>
      val t1 = STen.eye(3, STenOptions.d)
      assert(
        t1.cat(t1, 0).shape == List(6, 3)
      )
    }
  }
  test("cast to long") {
    Scope.root { implicit scope =>
      val t1 = STen.eye(2, STenOptions.d)
      assert(
        t1.castToLong.toLongMat == mat.ident(2).map(_.toLong)
      )
    }
  }
  test("cast to double") {
    Scope.root { implicit scope =>
      val t1 = STen.eye(2, STenOptions.l)
      assert(
        t1.castToDouble.toMat == mat.ident(2)
      )
    }
  }
  test("+") {
    Scope.root { implicit scope =>
      val t1 = STen.eye(2, STenOptions.d)
      assert(
        (t1 + t1 + t1) equalDeep (t1 * 3)
      )
    }
  }
  test("add") {
    Scope.root { implicit scope =>
      val t1 = STen.eye(2, STenOptions.d)
      assert(
        (t1.add(t1, 2d)) equalDeep (t1 * 3)
      )
    }
  }
  test("add2") {
    Scope.root { implicit scope =>
      val t1 = STen.zeros(List(2, 2), STenOptions.d)
      assert(
        (t1.add(1d, 1d)) equalDeep (STen.ones(List(2, 2), STenOptions.d))
      )
    }
  }
  test("+-") {
    Scope.root { implicit scope =>
      val t1 = STen.eye(2, STenOptions.d)
      assert(
        (t1 + t1 - t1) equalDeep (t1)
      )
    }
  }
  test("sub") {
    Scope.root { implicit scope =>
      val t1 = STen.ones(List(2, 2), STenOptions.d)
      assert(
        ((t1 + t1) sub (t1, 2d)) equalDeep (STen
          .zeros(List(2, 2), STenOptions.d))
      )
    }
  }
  test("*") {
    Scope.root { implicit scope =>
      val t1 = STen.ones(List(2, 2), STenOptions.d)
      val t0 = STen.zeros(List(2, 2), STenOptions.d)
      assert(
        (t1 * t0) equalDeep t0
      )
    }
  }
  test("/") {
    Scope.root { implicit scope =>
      val t1 = STen.ones(List(2, 2), STenOptions.d)
      val t2 = STen.ones(List(2, 2), STenOptions.d) * 2
      assert(
        (t1 / t2).sum.toMat.raw(0, 0) == 2
      )
    }
  }
  test("/2") {
    Scope.root { implicit scope =>
      val t1 = STen.ones(List(2, 2), STenOptions.d)
      assert(
        (t1 / 2).sum.toMat.raw(0, 0) == 2
      )
    }
  }
  test("mm") {
    Scope.root { implicit scope =>
      val t1 = STen.rand(List(2, 2), STenOptions.d)
      val t2 = STen.rand(List(2, 2), STenOptions.d)
      assert(
        (t1 mm t2).toMat.roundTo(4) == (t1.toMat mm t2.toMat).roundTo(4)
      )
    }
  }
  test("bmm") {
    Scope.root { implicit scope =>
      val t1 = STen.rand(List(1, 2, 2), STenOptions.d)
      val t2 = STen.rand(List(1, 2, 2), STenOptions.d)
      assert(
        (t1 bmm t2)
          .view(2, 2)
          .toMat == (t1.view(2, 2).toMat mm t2.view(2, 2).toMat)
      )
    }
  }
  test("relu") {
    Scope.root { implicit scope =>
      val t1 = STen.eye(3, STenOptions.d).neg
      assert(
        t1.relu equalDeep (STen.zeros(List(3, 3)))
      )
    }
  }
  test("exp log") {
    Scope.root { implicit scope =>
      val t1 = STen.ones(List(3, 3))
      assert(
        t1.exp.log.round equalDeep t1
      )
    }
  }
  test("square sqrt") {
    Scope.root { implicit scope =>
      val t1 = STen.ones(List(3, 3)) + 1
      assert(
        t1.square.sqrt.round equalDeep t1
      )
    }
  }
  test("abs") {
    Scope.root { implicit scope =>
      val t1 = STen.ones(List(3, 3)).neg
      assert(
        t1.abs equalDeep t1.neg
      )
    }
  }
  test("floor") {
    Scope.root { implicit scope =>
      val t1 = STen.ones(List(3, 3)) + 0.7
      assert(
        t1.floor equalDeep STen.ones(List(3, 3))
      )
    }
  }
  test("ceil") {
    Scope.root { implicit scope =>
      val t1 = STen.ones(List(3, 3)) + 0.7
      assert(
        t1.ceil equalDeep (STen.ones(List(3, 3)) + 1)
      )
    }
  }
  test("reciprocal") {
    Scope.root { implicit scope =>
      val t1 = STen.rand(List(3, 3))
      assert(
        t1.reciprocal equalDeep t1.pow(-1)
      )
    }
  }
  test("trace") {
    Scope.root { implicit scope =>
      val t1 = STen.eye(3)
      assert(
        t1.trace.toMat.raw(0) == 3d
      )
    }
  }

}
