package lamp.autograd

import org.saddle._
import org.saddle.ops.BinOps._
import org.saddle.linalg._
import org.scalatest.funsuite.AnyFunSuite
import aten.ATen
import aten.TensorOptions
import lamp.nn.CudaTest
import lamp.syntax
import cats.effect.IO
import lamp.util.NDArray

class GradientSuite extends AnyFunSuite {

  val mat2x3 = Mat(Vec(1d, 2d), Vec(3d, 4d), Vec(5d, 6d))
  val ndx1 = NDArray(Array(1d), List(1))
  val ndx2 = NDArray(Array(1d, 1d), List(2))
  val nd1x2x3 = NDArray(mat2x3.toArray, List(1, 2, 3))
  val nd1x2x3x3 =
    NDArray((0 until 18).toArray.map(_.toDouble), List(1, 2, 3, 3))
  val nd1x2x2 = NDArray(mat.ones(2, 2).toArray, List(1, 2, 2))
  val nd1x2x2x2 = NDArray(mat.ones(2, 4).toArray, List(1, 2, 2, 2))
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
  def diffND(
      m: NDArray[Double]
  )(f: NDArray[Double] => Double): NDArray[Double] = {
    val eps = 1e-6

    NDArray.zeros(m.shape).mapWithIndex {
      case (_, idx) =>
        val epsM = NDArray.zeros(m.shape)
        epsM.set(idx, eps)
        val a = f(m + epsM)
        val b = f(m - epsM)
        val r = (a - b) / (2 * eps)
        r
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
        Vec(fun(m, false, false)._1).roundTo(4) == Vec(expectedValue).roundTo(
          4
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
  def testGradientAndValueND(
      id: String
  )(m: NDArray[Double], expectedValue: Double)(
      fun: (NDArray[Double], Boolean, Boolean) => (
          Double,
          Option[NDArray[Double]]
      )
  ) = {
    test(id + ": gradient is correct") {

      def diffNum(m: NDArray[Double]) = diffND(m)(m => fun(m, false, false)._1)
      def diffAuto(m: NDArray[Double]) = {
        fun(m, true, false)._2.get
      }
      assert(
        Vec(fun(m, false, false)._1).roundTo(4) == Vec(expectedValue).roundTo(
          4
        )
      )

      assert(diffAuto(m).toVec.roundTo(4) == diffNum(m).toVec.roundTo(4))
    }
    test(id + "/CUDA: gradient is correct", CudaTest) {

      def diffNum(m: NDArray[Double]) = diffND(m)(m => fun(m, false, true)._1)
      def diffAuto(m: NDArray[Double]) = {
        fun(m, true, true)._2.get
      }
      assert(
        Vec(fun(m, false, true)._1).roundTo(10) == Vec(expectedValue).roundTo(
          10
        )
      )

      assert(diffAuto(m).toVec.roundTo(4) == diffNum(m).toVec.roundTo(4))
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
  testGradientAndValue("gelu")(mat2x3_2, 16d) { (m, doBackprop, cuda) =>
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
  testGradientAndValue("tanh")(mat2x3_2, 2.1981) { (m, doBackprop, cuda) =>
    val x1 = param(TensorHelpers.fromMat(m, cuda))
    val L = x1.tanh.sum
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
  testGradientAndValue("mean")(mat2x3_2, 5.5d) { (m, doBackprop, cuda) =>
    val x1 = param(TensorHelpers.fromMat(m, cuda))
    val L = x1.mean(List(0)).sum
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
  testGradientAndValue("l2 logistic regression loss - nll_loss")(
    mat2x3_2,
    151.0000008318073
  ) { (m, doBackprop, cuda) =>
    val w = param(TensorHelpers.fromMat(m))
    val data = const(TensorHelpers.fromMat(mat3x2))
    val y =
      const(ATen.squeeze_0(TensorHelpers.fromLongMat(Mat(Vec(0L, 1L, 2L)))))
    val L =
      ((data.mm(w)).logSoftMax.nllLoss(y.value, 3, Sum) + w.squaredFrobenius)
    if (doBackprop) {
      L.backprop()
    }
    (
      TensorHelpers.toMat(L.value).raw(0),
      w.partialDerivative.map(t => TensorHelpers.toMat(t))
    )
  }
  testGradientAndValue("l2 logistic regression loss - nll_loss unreduced")(
    mat2x3_2,
    151.0000008318073
  ) { (m, doBackprop, cuda) =>
    val w = param(TensorHelpers.fromMat(m))
    val data = const(TensorHelpers.fromMat(mat3x2))
    val y =
      const(ATen.squeeze_0(TensorHelpers.fromLongMat(Mat(Vec(0L, 1L, 2L)))))
    val L =
      ((data
        .mm(w))
        .logSoftMax
        .nllLoss(y.value, 3, NoReduction)
        .sum + w.squaredFrobenius)
    if (doBackprop) {
      L.backprop()
    }
    (
      TensorHelpers.toMat(L.value).raw(0),
      w.partialDerivative.map(t => TensorHelpers.toMat(t))
    )
  }
  testGradientAndValue("weight norm - wrt g")(mat2x3.row(Array(0)), 12.7279) {
    (m, doBackprop, cuda) =>
      val v = param(TensorHelpers.fromMat(mat.ones(2, 3), cuda))
      val g = param(TensorHelpers.fromMat(m, cuda))
      val L = WeightNorm(v, g, 0).value.sum
      if (doBackprop) {
        L.backprop()
      }
      (
        TensorHelpers.toMat(L.value).raw(0),
        g.partialDerivative.map(t => TensorHelpers.toMat(t))
      )
  }
  testGradientAndValue("weight norm - wrt v")(mat2x3, 4.1500) {
    (m, doBackprop, cuda) =>
      val v = param(TensorHelpers.fromMat(m, cuda))
      val g = param(TensorHelpers.fromMat(mat.ones(1, 3), cuda))
      val L = WeightNorm(v, g, 0).value.sum
      if (doBackprop) {
        L.backprop()
      }
      (
        TensorHelpers.toMat(L.value).raw(0),
        v.partialDerivative.map(t => TensorHelpers.toMat(t))
      )
  }
  testGradientAndValueND("conv1d - wrt weights")(nd1x2x2, 30d) {
    (m, doBackprop, cuda) =>
      val input =
        param(NDArray.tensorFromNDArray(nd1x2x3, cuda))
      val weight = param(NDArray.tensorFromNDArray(m, cuda))

      val bias = param(TensorHelpers.fromVec(vec.ones(1), cuda))
      val output =
        Conv1D(input, weight, bias, 1, 0, 1, 1L).value

      val L = output.sum
      if (doBackprop) {
        L.backprop()
      }
      (
        TensorHelpers.toMat(L.value).raw(0),
        weight.partialDerivative.map(t => NDArray.tensorToNDArray(t))
      )
  }
  testGradientAndValueND("conv1d - wrt input")(nd1x2x3, 30d) {
    (m, doBackprop, cuda) =>
      val input =
        param(NDArray.tensorFromNDArray(m, cuda))
      val weight = param(NDArray.tensorFromNDArray(nd1x2x2, cuda))

      val bias = param(TensorHelpers.fromVec(vec.ones(1), cuda))
      val output =
        Conv1D(input, weight, bias, 1, 0, 1, 1L).value

      val L = output.sum
      if (doBackprop) {
        L.backprop()
      }
      (
        TensorHelpers.toMat(L.value).raw(0),
        input.partialDerivative.map(t => NDArray.tensorToNDArray(t))
      )
  }
  testGradientAndValueND("conv1d - padded - wrt weights")(nd1x2x2, 46d) {
    (m, doBackprop, cuda) =>
      val input =
        param(NDArray.tensorFromNDArray(nd1x2x3, cuda))
      val weight = param(NDArray.tensorFromNDArray(m, cuda))

      val bias = param(TensorHelpers.fromVec(vec.ones(1), cuda))
      val output =
        Conv1D(input, weight, bias, 1L, 1L, 1L, 1L).value

      val L = output.sum
      if (doBackprop) {
        L.backprop()
      }
      (
        TensorHelpers.toMat(L.value).raw(0),
        weight.partialDerivative.map(t => NDArray.tensorToNDArray(t))
      )
  }
  testGradientAndValueND("conv1d -padded - wrt input")(nd1x2x3, 46d) {
    (m, doBackprop, cuda) =>
      val input =
        param(NDArray.tensorFromNDArray(m, cuda))
      val weight = param(NDArray.tensorFromNDArray(nd1x2x2, cuda))

      val bias = param(TensorHelpers.fromVec(vec.ones(1), cuda))
      val output =
        Conv1D(input, weight, bias, 1, 1, 1, 1L).value

      val L = output.sum
      if (doBackprop) {
        L.backprop()
      }
      (
        TensorHelpers.toMat(L.value).raw(0),
        input.partialDerivative.map(t => NDArray.tensorToNDArray(t))
      )
  }
  testGradientAndValueND("conv1d - stride-2 - wrt weights")(nd1x2x2, 23d) {
    (m, doBackprop, cuda) =>
      val input =
        param(NDArray.tensorFromNDArray(nd1x2x3, cuda))
      val weight = param(NDArray.tensorFromNDArray(m, cuda))

      val bias = param(TensorHelpers.fromVec(vec.ones(1), cuda))
      val output =
        Conv1D(input, weight, bias, 2L, 1L, 1L, 1L).value

      val L = output.sum
      if (doBackprop) {
        L.backprop()
      }
      (
        TensorHelpers.toMat(L.value).raw(0),
        weight.partialDerivative.map(t => NDArray.tensorToNDArray(t))
      )
  }
  testGradientAndValueND("conv1d -stride-2 - wrt input")(nd1x2x3, 23d) {
    (m, doBackprop, cuda) =>
      val input =
        param(NDArray.tensorFromNDArray(m, cuda))
      val weight = param(NDArray.tensorFromNDArray(nd1x2x2, cuda))

      val bias = param(TensorHelpers.fromVec(vec.ones(1), cuda))
      val output =
        Conv1D(input, weight, bias, 2, 1, 1, 1L).value

      val L = output.sum
      if (doBackprop) {
        L.backprop()
      }
      (
        TensorHelpers.toMat(L.value).raw(0),
        input.partialDerivative.map(t => NDArray.tensorToNDArray(t))
      )
  }
  testGradientAndValueND("conv1d -stride-2 - wrt bias")(ndx1, 23d) {
    (m, doBackprop, cuda) =>
      val input =
        param(NDArray.tensorFromNDArray(nd1x2x3, cuda))
      val weight = param(NDArray.tensorFromNDArray(nd1x2x2, cuda))

      val bias = param(NDArray.tensorFromNDArray(m, cuda))
      val output =
        Conv1D(input, weight, bias, 2, 1, 1, 1L).value

      val L = output.sum
      if (doBackprop) {
        L.backprop()
      }
      (
        TensorHelpers.toMat(L.value).raw(0),
        bias.partialDerivative.map(t => NDArray.tensorToNDArray(t))
      )
  }
  testGradientAndValueND("conv2d - wrt weights")(nd1x2x2x2, 276d) {
    (m, doBackprop, cuda) =>
      val input =
        param(NDArray.tensorFromNDArray(nd1x2x3x3, cuda))
      val weight = param(NDArray.tensorFromNDArray(m, cuda))

      val bias = param(TensorHelpers.fromVec(vec.ones(1), cuda))
      val output =
        Conv2D(input, weight, bias, 1, 0, 1, 1L).value

      val L = output.sum
      if (doBackprop) {
        L.backprop()
      }
      (
        TensorHelpers.toMat(L.value).raw(0),
        weight.partialDerivative.map(t => NDArray.tensorToNDArray(t))
      )
  }
  testGradientAndValueND("conv2d - wrt input")(nd1x2x3x3, 276d) {
    (m, doBackprop, cuda) =>
      val input =
        param(NDArray.tensorFromNDArray(m, cuda))
      val weight = param(NDArray.tensorFromNDArray(nd1x2x2x2, cuda))

      val bias = param(TensorHelpers.fromVec(vec.ones(1), cuda))
      val output =
        Conv2D(input, weight, bias, 1, 0, 1, 1L).value

      val L = output.sum
      if (doBackprop) {
        L.backprop()
      }
      (
        TensorHelpers.toMat(L.value).raw(0),
        input.partialDerivative.map(t => NDArray.tensorToNDArray(t))
      )
  }
  testGradientAndValueND("conv2d - padded - wrt input")(nd1x2x3x3, 628d) {
    (m, doBackprop, cuda) =>
      val input =
        param(NDArray.tensorFromNDArray(m, cuda))
      val weight = param(NDArray.tensorFromNDArray(nd1x2x2x2, cuda))

      val bias = param(TensorHelpers.fromVec(vec.ones(1), cuda))
      val output =
        Conv2D(input, weight, bias, 1, 1, 1, 1L).value

      val L = output.sum
      if (doBackprop) {
        L.backprop()
      }
      (
        TensorHelpers.toMat(L.value).raw(0),
        input.partialDerivative.map(t => NDArray.tensorToNDArray(t))
      )
  }
  testGradientAndValueND("conv2d - wrt bias")(ndx1, 276d) {
    (m, doBackprop, cuda) =>
      val input =
        param(NDArray.tensorFromNDArray(nd1x2x3x3, cuda))
      val weight = param(NDArray.tensorFromNDArray(nd1x2x2x2, cuda))

      val bias = param(NDArray.tensorFromNDArray(m, cuda))
      val output =
        Conv2D(input, weight, bias, 1, 0, 1, 1L).value

      val L = output.sum
      if (doBackprop) {
        L.backprop()
      }
      (
        TensorHelpers.toMat(L.value).raw(0),
        bias.partialDerivative.map(t => NDArray.tensorToNDArray(t))
      )
  }

  testGradientAndValueND("maxpool1d padded")(nd1x2x3, 32d) {
    (m, doBackprop, cuda) =>
      val input =
        param(NDArray.tensorFromNDArray(m, cuda))
      val output =
        MaxPool1D(input, 2, 1, 1, 1).value

      val L = output.sum
      if (doBackprop) {
        L.backprop()
      }
      (
        TensorHelpers.toMat(L.value).raw(0),
        input.partialDerivative.map(t => NDArray.tensorToNDArray(t))
      )
  }
  testGradientAndValueND("maxpool1d unpadded")(nd1x2x3, 18d) {
    (m, doBackprop, cuda) =>
      val input =
        param(NDArray.tensorFromNDArray(m, cuda))

      val output =
        MaxPool1D(input, 2, 1, 0, 1).value

      val L = output.sum
      if (doBackprop) {
        L.backprop()
      }
      (
        TensorHelpers.toMat(L.value).raw(0),
        input.partialDerivative.map(t => NDArray.tensorToNDArray(t))
      )
  }
  testGradientAndValueND("maxpool1d strided")(nd1x2x3, 7d) {
    (m, doBackprop, cuda) =>
      val input =
        param(NDArray.tensorFromNDArray(m, cuda))

      val output =
        MaxPool1D(input, 2, 2, 0, 1).value

      val L = output.sum
      if (doBackprop) {
        L.backprop()
      }
      (
        TensorHelpers.toMat(L.value).raw(0),
        input.partialDerivative.map(t => NDArray.tensorToNDArray(t))
      )
  }
  testGradientAndValueND("maxpool2d strided")(nd1x2x3x3, 17d) {
    (m, doBackprop, cuda) =>
      val input =
        param(NDArray.tensorFromNDArray(m, cuda))

      val output =
        MaxPool2D(input, 2, 2, 0, 1).value

      val L = output.sum
      if (doBackprop) {
        L.backprop()
      }
      (
        TensorHelpers.toMat(L.value).raw(0),
        input.partialDerivative.map(t => NDArray.tensorToNDArray(t))
      )
  }
  testGradientAndValueND("maxpool2d strided padded")(nd1x2x3x3, 68d) {
    (m, doBackprop, cuda) =>
      val input =
        param(NDArray.tensorFromNDArray(m, cuda))

      val output =
        MaxPool2D(input, 2, 2, 1, 1).value

      val L = output.sum
      if (doBackprop) {
        L.backprop()
      }
      (
        TensorHelpers.toMat(L.value).raw(0),
        input.partialDerivative.map(t => NDArray.tensorToNDArray(t))
      )
  }
  testGradientAndValueND("avgpool2d strided padded")(nd1x2x3x3, 38.25) {
    (m, doBackprop, cuda) =>
      val input =
        param(NDArray.tensorFromNDArray(m, cuda))

      val output =
        AvgPool2D(input, 2, 2, 1).value

      val L = output.sum
      if (doBackprop) {
        L.backprop()
      }
      (
        TensorHelpers.toMat(L.value).raw(0),
        input.partialDerivative.map(t => NDArray.tensorToNDArray(t))
      )
  }
}
