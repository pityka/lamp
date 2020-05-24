package candle.autograd

import org.saddle._
import org.saddle.ops.BinOps._
import org.saddle.linalg._
import org.scalatest.funsuite.AnyFunSuite

class GradientSuite extends AnyFunSuite {

  val mat2x3 = Mat(Vec(1d, 2d), Vec(3d, 4d), Vec(5d, 6d))
  val mat2x3_2 = Mat(Vec(-1d, 2d), Vec(3d, -4d), Vec(5d, 6d))
  val mat3x2 = mat2x3.T

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
      fun: (Mat[Double], Boolean) => (Double, Option[Mat[Double]])
  ) = test(id + ": gradient is correct") {

    def diffNum(m: Mat[Double]) = diff(m)(m => fun(m, false)._1)
    def diffAuto(m: Mat[Double]) = {
      fun(m, true)._2.get
    }
    assert(fun(m, false)._1 == expectedValue)

    assert(diffAuto(m).roundTo(4) == diffNum(m).roundTo(4))
  }

  test("constant is not accumulating gradients") {
    val x1 = const(mat2x3)
    val L = x1.sum
    assert(L.value == Mat(Vec(mat2x3.toVec.sum2)))
    L.backprop()
    assert(x1.partialDerivative.isEmpty)
  }
  test("param is accumulating gradients") {
    val x1 = param(mat2x3)
    val L = x1.sum
    assert(L.value == Mat(Vec(mat2x3.toVec.sum2)))
    L.backprop()
    assert(x1.partialDerivative.get == mat.ones(2, 3))
  }

  testGradientAndValue("sum")(mat2x3, 21d) { (m, doBackprop) =>
    val x1 = param(m)
    val L = x1.sum
    if (doBackprop) {
      L.backprop()
    }
    (L.value.raw(0), x1.partialDerivative)
  }
  testGradientAndValue("colSum")(mat2x3, 21d) { (m, doBackprop) =>
    val x1 = param(m)
    val L = x1.colSum.sum
    if (doBackprop) {
      L.backprop()
    }
    (L.value.raw(0), x1.partialDerivative)
  }
  testGradientAndValue("rowSum")(mat2x3, 21d) { (m, doBackprop) =>
    val x1 = param(m)
    val L = x1.rowSum.sum
    if (doBackprop) {
      L.backprop()
    }
    (L.value.raw(0), x1.partialDerivative)
  }
  testGradientAndValue("add - left")(mat2x3, 63d) { (m, doBackprop) =>
    val x1 = param(m)
    val x2 = param(mat2x3 * 2)
    val L = x1.+(x2).sum
    if (doBackprop) {
      L.backprop()
    }
    (L.value.raw(0), x1.partialDerivative)
  }
  testGradientAndValue("add - right")(mat2x3, 63d) { (m, doBackprop) =>
    val x1 = param(m)
    val x2 = param(mat2x3 * 2)
    val L = x2.+(x1).sum
    if (doBackprop) {
      L.backprop()
    }
    (L.value.raw(0), x1.partialDerivative)
  }
  testGradientAndValue("minus - left")(mat2x3, -21d) { (m, doBackprop) =>
    val x1 = param(m)
    val x2 = param(mat2x3 * 2)
    val L = x1.-(x2).sum
    if (doBackprop) {
      L.backprop()
    }
    (L.value.raw(0), x1.partialDerivative)
  }
  testGradientAndValue("minus - right")(mat2x3, 21d) { (m, doBackprop) =>
    val x1 = param(m)
    val x2 = param(mat2x3 * 2)
    val L = x2.-(x1).sum
    if (doBackprop) {
      L.backprop()
    }
    (L.value.raw(0), x1.partialDerivative)
  }
  testGradientAndValue("mult - left")(mat2x3, 182d) { (m, doBackprop) =>
    val x1 = param(m)
    val x2 = param(mat2x3 * 2)
    val L = x1.*(x2).sum
    if (doBackprop) {
      L.backprop()
    }
    (L.value.raw(0), x1.partialDerivative)
  }
  testGradientAndValue("mult - right")(mat2x3, 182d) { (m, doBackprop) =>
    val x1 = param(m)
    val x2 = param(mat2x3 * 2)
    val L = x2.*(x1).sum
    if (doBackprop) {
      L.backprop()
    }
    (L.value.raw(0), x1.partialDerivative)
  }

  testGradientAndValue("div - left")(mat2x3, 3d) { (m, doBackprop) =>
    val x1 = param(m)
    val x2 = param(mat2x3 * 2)
    val L = x1./(x2).sum
    if (doBackprop) {
      L.backprop()
    }
    (L.value.raw(0), x1.partialDerivative)
  }
  testGradientAndValue("div - right")(mat2x3, 12d) { (m, doBackprop) =>
    val x1 = param(m)
    val x2 = param(mat2x3 * 2)
    val L = x2./(x1).sum
    if (doBackprop) {
      L.backprop()
    }
    (L.value.raw(0), x1.partialDerivative)
  }

  testGradientAndValue("mm - left")(mat2x3, 358d) { (m, doBackprop) =>
    val x1 = param(m)
    val x2 = param(mat3x2 * 2)
    val L = x1.mm(x2).sum
    if (doBackprop) {
      L.backprop()
    }
    (L.value.raw(0), x1.partialDerivative)
  }
  testGradientAndValue("mm - right")(mat2x3, 450d) { (m, doBackprop) =>
    val x1 = param(m)
    val x2 = param(mat3x2 * 2)
    val L = x2.mm(x1).sum
    if (doBackprop) {
      L.backprop()
    }
    (L.value.raw(0), x1.partialDerivative)
  }
  testGradientAndValue("crossentropy - left")(mat2x3, -182.0) {
    (m, doBackprop) =>
      val x1 = param(m)
      val x2 = param(mat2x3 * 2)
      val L = x1.crossEntropy(x2).sum
      if (doBackprop) {
        L.backprop()
      }
      (L.value.raw(0), x1.partialDerivative)
  }
  testGradientAndValue("crossentropy - right")(mat2x3, -182.0) {
    (m, doBackprop) =>
      val x1 = param(m)
      val x2 = param(mat2x3 * 2)
      val L = x2.crossEntropy(x1).sum
      if (doBackprop) {
        L.backprop()
      }
      (L.value.raw(0), x1.partialDerivative)
  }

  testGradientAndValue("relu")(mat2x3_2, 16d) { (m, doBackprop) =>
    val x1 = param(m)
    val L = x1.relu.sum
    if (doBackprop) {
      L.backprop()
    }
    (L.value.raw(0), x1.partialDerivative)
  }
  testGradientAndValue("exp")(mat2x3_2, 579.7027406974902) { (m, doBackprop) =>
    val x1 = param(m)
    val L = x1.exp.sum
    if (doBackprop) {
      L.backprop()
    }
    (L.value.raw(0), x1.partialDerivative)
  }
  testGradientAndValue("log")(mat2x3, 6.579251212010101) { (m, doBackprop) =>
    val x1 = param(m)
    val L = x1.log.sum
    if (doBackprop) {
      L.backprop()
    }
    (L.value.raw(0), x1.partialDerivative)
  }
  testGradientAndValue("sin")(mat2x3_2, -0.2725908274764838) {
    (m, doBackprop) =>
      val x1 = param(m)
      val L = x1.sin.sum
      if (doBackprop) {
        L.backprop()
      }
      (L.value.raw(0), x1.partialDerivative)
  }
  testGradientAndValue("cos")(mat2x3_2, -0.2756481760294678) {
    (m, doBackprop) =>
      val x1 = param(m)
      val L = x1.cos.sum
      if (doBackprop) {
        L.backprop()
      }
      (L.value.raw(0), x1.partialDerivative)
  }
  testGradientAndValue("tan")(mat2x3_2, -8.714336610971612) { (m, doBackprop) =>
    val x1 = param(m)
    val L = x1.tan.sum
    if (doBackprop) {
      L.backprop()
    }
    (L.value.raw(0), x1.partialDerivative)
  }
  testGradientAndValue("atan")(mat2x3_2, 3.02402707945215) { (m, doBackprop) =>
    val x1 = param(m)
    val L = x1.atan.sum
    if (doBackprop) {
      L.backprop()
    }
    (L.value.raw(0), x1.partialDerivative)
  }
  testGradientAndValue("pow")(mat2x3_2, 91d) { (m, doBackprop) =>
    val x1 = param(m)
    val L = x1.pow(2d).sum
    if (doBackprop) {
      L.backprop()
    }
    (L.value.raw(0), x1.partialDerivative)
  }
  testGradientAndValue("softmax")(mat2x3_2, -22.441910257332836) {
    (m, doBackprop) =>
      val x1 = param(m)
      val L = x1.logSoftMax.sum
      if (doBackprop) {
        L.backprop()
      }
      (L.value.raw(0), x1.partialDerivative)
  }
  testGradientAndValue("squaredFrobenius")(mat2x3_2, 91d) { (m, doBackprop) =>
    val x1 = param(m)
    val L = x1.squaredFrobenius.sum
    if (doBackprop) {
      L.backprop()
    }
    (L.value.raw(0), x1.partialDerivative)
  }
  testGradientAndValue("l2 logistic regression loss")(
    mat2x3_2,
    151.0000008318073
  ) { (m, doBackprop) =>
    val w = param(m)
    val data = const(mat3x2)
    val y = const(mat.ident(3))
    val L =
      ((data.mm(w)).logSoftMax.crossEntropy(y).sum + w.squaredFrobenius)
    if (doBackprop) {
      L.backprop()
    }
    (L.value.raw(0), w.partialDerivative)
  }

}
