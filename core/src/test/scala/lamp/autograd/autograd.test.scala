package lamp.autograd

import org.saddle._
import org.saddle.ops.BinOps._
import org.saddle.linalg._
import org.scalatest.funsuite.AnyFunSuite
import aten.ATen
import aten.TensorOptions
import lamp.nn.CudaTest

class GradientSuite extends AnyFunSuite {

  val mat2x3 = Mat(Vec(1d, 2d), Vec(3d, 4d), Vec(5d, 6d))
  val mat1x1 = Mat(Vec(1d))
  val mat3x2 = mat2x3.T
  val t2x3 = TensorHelpers.fromMat(mat2x3)
  val mat2x3_2 = Mat(Vec(-1d, 2d), Vec(3d, -4d), Vec(5d, 6d))
  val t3x2 = ATen.t(t2x3)

  def diff(m: Mat[Double])(f: Mat[Double] => Double): Mat[Double] = {
    val eps = 1e-6
    mat.zeros(m.numRows, m.numCols).mapRows {
      case (row, i) =>
        (0 until row.length).map { j =>
          val epsM = mat.zeros(m.numRows, m.numCols)
          epsM(i, j) = eps
          (f(m + epsM) - f(m - epsM)) / (2 * eps)
        }.toVec
    }

  }

  def testGradientAndValue(id: String)(m: Mat[Double], expectedValue: Double)(
      fun: (Mat[Double], Boolean, Boolean) => (Double, Option[Mat[Double]])
  ) = {
    test(id + ": gradient is correct") {

      def diffNum(m: Mat[Double]) = diff(m)(m => fun(m, false, false)._1)
      def diffAuto(m: Mat[Double]) = {
        fun(m, true, false)._2.get
      }
      assert(
        Vec(fun(m, false, false)._1).roundTo(10) == Vec(expectedValue).roundTo(
          10
        )
      )

      assert(diffAuto(m).roundTo(4) == diffNum(m).roundTo(4))
    }
    test(id + "/CUDA: gradient is correct", CudaTest) {

      def diffNum(m: Mat[Double]) = diff(m)(m => fun(m, false, true)._1)
      def diffAuto(m: Mat[Double]) = {
        fun(m, true, true)._2.get
      }
      assert(
        Vec(fun(m, false, true)._1).roundTo(10) == Vec(expectedValue).roundTo(
          10
        )
      )

      assert(diffAuto(m).roundTo(4) == diffNum(m).roundTo(4))
    }
  }

  test("constant is not accumulating gradients") {
    val x1 = const(t2x3)
    val L = x1.sum
    assert(
      TensorHelpers
        .toMat(L.value) == Mat(Vec(TensorHelpers.toMat(t2x3).toVec.sum2))
    )
    L.backprop()
    assert(x1.partialDerivative.isEmpty)
  }
  test("param is accumulating gradients") {
    val x1 = param(t2x3)
    val L = x1.sum
    // assert(L.value == Mat(Vec(mat2x3.toVec.sum2)))
    L.backprop()
    assert(TensorHelpers.toMat(x1.partialDerivative.get) == mat.ones(2, 3))
  }

  testGradientAndValue("sum")(mat2x3, 21d) { (m, doBackprop, cuda) =>
    val x1 = param(TensorHelpers.fromMat(m, cuda))
    val L = x1.sum
    if (doBackprop) {
      L.backprop()
    }
    (
      TensorHelpers.toMat(L.value).raw(0),
      x1.partialDerivative.map(t => TensorHelpers.toMat(t))
    )
  }
  testGradientAndValue("colSum")(mat2x3, 21d) { (m, doBackprop, cuda) =>
    val x1 = param(TensorHelpers.fromMat(m, cuda))
    val L = x1.colSum.sum
    if (doBackprop) {
      L.backprop()
    }
    (
      TensorHelpers.toMat(L.value).raw(0),
      x1.partialDerivative.map(t => TensorHelpers.toMat(t))
    )
  }
  testGradientAndValue("rowSum")(mat2x3, 21d) { (m, doBackprop, cuda) =>
    val x1 = param(TensorHelpers.fromMat(m, cuda))
    val L = x1.rowSum.sum
    if (doBackprop) {
      L.backprop()
    }
    (
      TensorHelpers.toMat(L.value).raw(0),
      x1.partialDerivative.map(t => TensorHelpers.toMat(t))
    )
  }
  testGradientAndValue("add broadcasted - left")(Mat(Vec(1d)), 48d) {
    (m, doBackprop, cuda) =>
      val x1 = param(TensorHelpers.fromMat(m, cuda))
      val x2 = param(TensorHelpers.fromMat(mat2x3 * 2, cuda))
      val L = (x1.+(x2)).sum
      if (doBackprop) {
        L.backprop()
      }
      (
        TensorHelpers.toMat(L.value).raw(0),
        x1.partialDerivative.map(t => TensorHelpers.toMat(t))
      )
  }
  testGradientAndValue("add - left")(mat2x3, 63d) { (m, doBackprop, cuda) =>
    val x1 = param(TensorHelpers.fromMat(m, cuda))
    val x2 = param(TensorHelpers.fromMat(mat2x3 * 2, cuda))
    val L = x1.+(x2).sum
    if (doBackprop) {
      L.backprop()
    }
    (
      TensorHelpers.toMat(L.value).raw(0),
      x1.partialDerivative.map(t => TensorHelpers.toMat(t))
    )
  }
  testGradientAndValue("add broadcasted - right")(Mat(Vec(1d)), 48d) {
    (m, doBackprop, cuda) =>
      val x1 = param(TensorHelpers.fromMat(m, cuda))
      val x2 = param(TensorHelpers.fromMat(mat2x3 * 2, cuda))
      val L = (x2.+(x1)).sum
      if (doBackprop) {
        L.backprop()
      }
      (
        TensorHelpers.toMat(L.value).raw(0),
        x1.partialDerivative.map(t => TensorHelpers.toMat(t))
      )
  }
  testGradientAndValue("add - right")(mat2x3, 63d) { (m, doBackprop, cuda) =>
    val x1 = param(TensorHelpers.fromMat(m, cuda))
    val x2 = param(TensorHelpers.fromMat(mat2x3 * 2, cuda))
    val L = x2.+(x1).sum
    if (doBackprop) {
      L.backprop()
    }
    (
      TensorHelpers.toMat(L.value).raw(0),
      x1.partialDerivative.map(t => TensorHelpers.toMat(t))
    )
  }
  testGradientAndValue("minus - left")(mat2x3, -21d) { (m, doBackprop, cuda) =>
    val x1 = param(TensorHelpers.fromMat(m, cuda))
    val x2 = param(TensorHelpers.fromMat(mat2x3 * 2, cuda))
    val L = x1.-(x2).sum
    if (doBackprop) {
      L.backprop()
    }
    (
      TensorHelpers.toMat(L.value).raw(0),
      x1.partialDerivative.map(t => TensorHelpers.toMat(t))
    )
  }
  testGradientAndValue("minus broadcasted - left")(mat1x1, -36d) {
    (m, doBackprop, cuda) =>
      val x1 = param(TensorHelpers.fromMat(m, cuda))
      val x2 = param(TensorHelpers.fromMat(mat2x3 * 2, cuda))
      val L = x1.-(x2).sum
      if (doBackprop) {
        L.backprop()
      }
      (
        TensorHelpers.toMat(L.value).raw(0),
        x1.partialDerivative.map(t => TensorHelpers.toMat(t))
      )
  }
  testGradientAndValue("minus broadcasted - right")(mat1x1, 36d) {
    (m, doBackprop, cuda) =>
      val x1 = param(TensorHelpers.fromMat(m, cuda))
      val x2 = param(TensorHelpers.fromMat(mat2x3 * 2, cuda))
      val L = x2.-(x1).sum
      if (doBackprop) {
        L.backprop()
      }
      (
        TensorHelpers.toMat(L.value).raw(0),
        x1.partialDerivative.map(t => TensorHelpers.toMat(t))
      )
  }
  testGradientAndValue("minus - right")(mat2x3, 21d) { (m, doBackprop, cuda) =>
    val x1 = param(TensorHelpers.fromMat(m, cuda))
    val x2 = param(TensorHelpers.fromMat(mat2x3 * 2, cuda))
    val L = x2.-(x1).sum
    if (doBackprop) {
      L.backprop()
    }
    (
      TensorHelpers.toMat(L.value).raw(0),
      x1.partialDerivative.map(t => TensorHelpers.toMat(t))
    )
  }
  testGradientAndValue("mult - left")(mat2x3, 182d) { (m, doBackprop, cuda) =>
    val x1 = param(TensorHelpers.fromMat(m, cuda))
    val x2 = param(TensorHelpers.fromMat(mat2x3 * 2, cuda))
    val L = x1.*(x2).sum
    if (doBackprop) {
      L.backprop()
    }
    (
      TensorHelpers.toMat(L.value).raw(0),
      x1.partialDerivative.map(t => TensorHelpers.toMat(t))
    )
  }
  testGradientAndValue("mult broadcasted - left")(mat1x1, 42d) {
    (m, doBackprop, cuda) =>
      val x1 = param(TensorHelpers.fromMat(m, cuda))
      val x2 = param(TensorHelpers.fromMat(mat2x3 * 2, cuda))
      val L = x1.*(x2).sum
      if (doBackprop) {
        L.backprop()
      }
      (
        TensorHelpers.toMat(L.value).raw(0),
        x1.partialDerivative.map(t => TensorHelpers.toMat(t))
      )
  }
  testGradientAndValue("mult broadcasted - right")(mat1x1, 42d) {
    (m, doBackprop, cuda) =>
      val x1 = param(TensorHelpers.fromMat(m, cuda))
      val x2 = param(TensorHelpers.fromMat(mat2x3 * 2, cuda))
      val L = x2.*(x1).sum
      if (doBackprop) {
        L.backprop()
      }
      (
        TensorHelpers.toMat(L.value).raw(0),
        x1.partialDerivative.map(t => TensorHelpers.toMat(t))
      )
  }
  testGradientAndValue("mult - right")(mat2x3, 182d) { (m, doBackprop, cuda) =>
    val x1 = param(TensorHelpers.fromMat(m, cuda))
    val x2 = param(TensorHelpers.fromMat(mat2x3 * 2, cuda))
    val L = x2.*(x1).sum
    if (doBackprop) {
      L.backprop()
    }
    (
      TensorHelpers.toMat(L.value).raw(0),
      x1.partialDerivative.map(t => TensorHelpers.toMat(t))
    )
  }

  testGradientAndValue("div - left")(mat2x3, 3d) { (m, doBackprop, cuda) =>
    val x1 = param(TensorHelpers.fromMat(m, cuda))
    val x2 = param(TensorHelpers.fromMat(mat2x3 * 2, cuda))
    val L = x1./(x2).sum
    if (doBackprop) {
      L.backprop()
    }
    (
      TensorHelpers.toMat(L.value).raw(0),
      x1.partialDerivative.map(t => TensorHelpers.toMat(t))
    )
  }
  testGradientAndValue("div broadcasted - left")(mat1x1, 1.225d) {
    (m, doBackprop, cuda) =>
      val x1 = param(TensorHelpers.fromMat(m, cuda))
      val x2 = param(TensorHelpers.fromMat(mat2x3 * 2, cuda))
      val L = x1./(x2).sum
      if (doBackprop) {
        L.backprop()
      }
      (
        TensorHelpers.toMat(L.value).raw(0),
        x1.partialDerivative.map(t => TensorHelpers.toMat(t))
      )
  }
  testGradientAndValue("div - right")(mat2x3, 12d) { (m, doBackprop, cuda) =>
    val x1 = param(TensorHelpers.fromMat(m, cuda))
    val x2 = param(TensorHelpers.fromMat(mat2x3 * 2, cuda))
    val L = x2./(x1).sum
    if (doBackprop) {
      L.backprop()
    }
    (
      TensorHelpers.toMat(L.value).raw(0),
      x1.partialDerivative.map(t => TensorHelpers.toMat(t))
    )
  }

  testGradientAndValue("mm - left")(mat2x3, 358d) { (m, doBackprop, cuda) =>
    val x1 = param(TensorHelpers.fromMat(m, cuda))
    val x2 = param(TensorHelpers.fromMat(mat3x2 * 2, cuda))
    val L = x1.mm(x2).sum
    if (doBackprop) {
      L.backprop()
    }
    (
      TensorHelpers.toMat(L.value).raw(0),
      x1.partialDerivative.map(t => TensorHelpers.toMat(t))
    )
  }
  testGradientAndValue("mm - right")(mat2x3, 450d) { (m, doBackprop, cuda) =>
    val x1 = param(TensorHelpers.fromMat(m, cuda))
    val x2 = param(TensorHelpers.fromMat(mat3x2 * 2, cuda))
    val L = x2.mm(x1).sum
    if (doBackprop) {
      L.backprop()
    }
    (
      TensorHelpers.toMat(L.value).raw(0),
      x1.partialDerivative.map(t => TensorHelpers.toMat(t))
    )
  }
  testGradientAndValue("crossentropy - left")(mat2x3, -182.0) {
    (m, doBackprop, cuda) =>
      val x1 = param(TensorHelpers.fromMat(m, cuda))
      val x2 = param(TensorHelpers.fromMat(mat2x3 * 2, cuda))

      val L = x1.crossEntropy(x2).sum
      if (doBackprop) {
        L.backprop()
      }
      (
        TensorHelpers.toMat(L.value).raw(0),
        x1.partialDerivative.map(t => TensorHelpers.toMat(t))
      )
  }
  testGradientAndValue("crossentropy - right")(mat2x3, -182.0) {
    (m, doBackprop, cuda) =>
      val x1 = param(TensorHelpers.fromMat(m, cuda))
      val x2 = param(TensorHelpers.fromMat(mat2x3 * 2, cuda))
      val L = x2.crossEntropy(x1).sum
      if (doBackprop) {
        L.backprop()
      }
      (
        TensorHelpers.toMat(L.value).raw(0),
        x1.partialDerivative.map(t => TensorHelpers.toMat(t))
      )
  }

  testGradientAndValue("relu")(mat2x3_2, 16d) { (m, doBackprop, cuda) =>
    val x1 = param(TensorHelpers.fromMat(m, cuda))
    val L = x1.relu.sum
    if (doBackprop) {
      L.backprop()
    }
    (
      TensorHelpers.toMat(L.value).raw(0),
      x1.partialDerivative.map(t => TensorHelpers.toMat(t))
    )
  }
  testGradientAndValue("exp")(mat2x3_2, 579.7027406974902) {
    (m, doBackprop, cuda) =>
      val x1 = param(TensorHelpers.fromMat(m, cuda))
      val L = x1.exp.sum
      if (doBackprop) {
        L.backprop()
      }
      (
        TensorHelpers.toMat(L.value).raw(0),
        x1.partialDerivative.map(t => TensorHelpers.toMat(t))
      )
  }
  testGradientAndValue("log")(mat2x3, 6.579251212010101) {
    (m, doBackprop, cuda) =>
      val x1 = param(TensorHelpers.fromMat(m, cuda))
      val L = x1.log.sum
      if (doBackprop) {
        L.backprop()
      }
      (
        TensorHelpers.toMat(L.value).raw(0),
        x1.partialDerivative.map(t => TensorHelpers.toMat(t))
      )
  }
  testGradientAndValue("sin")(mat2x3_2, -0.27259082747648367) {
    (m, doBackprop, cuda) =>
      val x1 = param(TensorHelpers.fromMat(m, cuda))
      val L = x1.sin.sum
      if (doBackprop) {
        L.backprop()
      }
      (
        TensorHelpers.toMat(L.value).raw(0),
        x1.partialDerivative.map(t => TensorHelpers.toMat(t))
      )
  }
  testGradientAndValue("cos")(mat2x3_2, -0.2756481760294678) {
    (m, doBackprop, cuda) =>
      val x1 = param(TensorHelpers.fromMat(m, cuda))
      val L = x1.cos.sum
      if (doBackprop) {
        L.backprop()
      }
      (
        TensorHelpers.toMat(L.value).raw(0),
        x1.partialDerivative.map(t => TensorHelpers.toMat(t))
      )
  }
  testGradientAndValue("tan")(mat2x3_2, -8.71433661097161) {
    (m, doBackprop, cuda) =>
      val x1 = param(TensorHelpers.fromMat(m, cuda))
      val L = x1.tan.sum
      if (doBackprop) {
        L.backprop()
      }
      (
        TensorHelpers.toMat(L.value).raw(0),
        x1.partialDerivative.map(t => TensorHelpers.toMat(t))
      )
  }
  testGradientAndValue("atan")(mat2x3_2, 3.02402707945215) {
    (m, doBackprop, cuda) =>
      val x1 = param(TensorHelpers.fromMat(m, cuda))
      val L = x1.atan.sum
      if (doBackprop) {
        L.backprop()
      }
      (
        TensorHelpers.toMat(L.value).raw(0),
        x1.partialDerivative.map(t => TensorHelpers.toMat(t))
      )
  }
  testGradientAndValue("pow")(mat2x3_2, 91d) { (m, doBackprop, cuda) =>
    val x1 = param(TensorHelpers.fromMat(m, cuda))
    val L = x1.pow(2d).sum
    if (doBackprop) {
      L.backprop()
    }
    (
      TensorHelpers.toMat(L.value).raw(0),
      x1.partialDerivative.map(t => TensorHelpers.toMat(t))
    )
  }
  testGradientAndValue("softmax")(mat2x3_2, -22.441910257332836) {
    (m, doBackprop, cuda) =>
      val x1 = param(TensorHelpers.fromMat(m, cuda))
      val L = x1.logSoftMax.sum
      if (doBackprop) {
        L.backprop()
      }
      (
        TensorHelpers.toMat(L.value).raw(0),
        x1.partialDerivative.map(t => TensorHelpers.toMat(t))
      )
  }
  testGradientAndValue("squaredFrobenius")(mat2x3_2, 91d) {
    (m, doBackprop, cuda) =>
      val x1 = param(TensorHelpers.fromMat(m, cuda))
      val L = x1.squaredFrobenius.sum
      if (doBackprop) {
        L.backprop()
      }
      (
        TensorHelpers.toMat(L.value).raw(0),
        x1.partialDerivative.map(t => TensorHelpers.toMat(t))
      )
  }
  testGradientAndValue("transpose")(mat2x3_2, 11d) { (m, doBackprop, cuda) =>
    val x1 = param(TensorHelpers.fromMat(m, cuda))
    val L = x1.t.sum
    if (doBackprop) {
      L.backprop()
    }
    (
      TensorHelpers.toMat(L.value).raw(0),
      x1.partialDerivative.map(t => TensorHelpers.toMat(t))
    )
  }
  testGradientAndValue("l2 logistic regression loss")(
    mat2x3_2,
    151.0000008318073
  ) { (m, doBackprop, cuda) =>
    val w = param(TensorHelpers.fromMat(m))
    val data = const(TensorHelpers.fromMat(mat3x2))
    val y = const(TensorHelpers.fromMat(mat.ident(3)))
    val L =
      ((data.mm(w)).logSoftMax.crossEntropy(y).sum + w.squaredFrobenius)
    if (doBackprop) {
      L.backprop()
    }
    (
      TensorHelpers.toMat(L.value).raw(0),
      w.partialDerivative.map(t => TensorHelpers.toMat(t))
    )
  }

}
