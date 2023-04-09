package lamp.nn

import org.scalatest.funsuite.AnyFunSuite

import lamp.nn.MultiheadAttention
import lamp._
import lamp.autograd.const
import lamp.saddle._
import org.saddle._

class MaskedSoftmaxTest extends AnyFunSuite {

  test("1D") {
    Scope.unsafe { implicit sc =>
      val maxLength = STen.fromLongArray(Array(2, 3))
      val maskable = STen.ones(List(2, 4, 3))
      val fill = 0.0

      val masked = MultiheadAttention
        .sequenceMaskValidLength1D(maxLength, const(maskable), fill)
        .value

      val b0 =
        masked
          .select(dim = 0, index = 0)
          .toMat
      val b1 =
        masked
          .select(dim = 0, index = 1)
          .toMat

      assert(b0 == Mat(vec.ones(4), vec.ones(4), vec.zeros(4)))
      assert(b1 == Mat(vec.ones(4), vec.ones(4), vec.ones(4)))

    }
  }

  test("1D symm") {
    Scope.unsafe { implicit sc =>
      val maxLength = STen.fromLongArray(Array(2, 3))
      val maskable = STen.ones(List(2, 4, 4))
      val fill = 0.0

      val masked = MultiheadAttention
        .sequenceMaskValidLength1D(maxLength, const(maskable), fill)
        .value

      val b0 =
        masked
          .select(dim = 0, index = 0)
          .toMat
      val b1 =
        masked
          .select(dim = 0, index = 1)
          .toMat

      assert(b0 == Mat(vec.ones(4), vec.ones(4), vec.zeros(4), vec.zeros(4)))
      assert(b1 == Mat(vec.ones(4), vec.ones(4), vec.ones(4), vec.zeros(4)))

    }
  }

  test("2D") {
    Scope.unsafe { implicit sc =>
      val maxLength = STen.fromLongArray(Array(2, 3, 2, 1)).reshape(2, 2)
      val maskable = STen.ones(List(2, 2, 3))
      val fill = 0.0

      val masked = MultiheadAttention
        .sequenceMaskValidLength2D(maxLength, const(maskable), fill)
        .value

      val b0 =
        masked
          .select(dim = 0, index = 0)
          .toMat

      val b1 =
        masked
          .select(dim = 0, index = 1)
          .toMat

      assert(b0 == Mat(Vec(1d, 1d, 0d), Vec(1d, 1d, 1d)).T)
      assert(b1 == Mat(Vec(1d, 1d, 0d), Vec(1d, 0d, 0d)).T)

    }
  }

}
