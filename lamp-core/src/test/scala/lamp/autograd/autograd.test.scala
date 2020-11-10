package lamp.autograd

import org.saddle._
import org.saddle.ops.BinOps._
import org.scalatest.funsuite.AnyFunSuite
import aten.ATen
import lamp.nn.CudaTest
import lamp.util.syntax
import lamp.Scope
import lamp.util.NDArray
import aten.Tensor
import aten.TensorOptions
import lamp.TensorHelpers

class GradientSuite extends AnyFunSuite {
  val ar18 = Array(1d, 2d, 3d, 4d, 5d, 6d, 1d, 2d, 3d, 4d, 5d, 6d, 1d, 2d, 3d,
    4d, 5d, 6d)
  val mat2x3 = Mat(Vec(1d, 2d), Vec(3d, 4d), Vec(5d, 6d))
  val ndx1 = NDArray(Array(1d), List(1))
  val ndx2 = NDArray(Array(1d, 1d), List(2))
  val ndx3 = NDArray(Array(1d, 2d, 3d), List(3))
  val ndx6 = NDArray(Array(1d, 2d, 3d, 4d, 5d, 6d), List(6))
  val ndx18 = NDArray(
    ar18,
    List(18)
  )
  val nd1x2x3 = NDArray(mat2x3.toArray, List(1, 2, 3))
  val nd1x2x3_2 = NDArray((mat2x3 * 3).toArray, List(1, 2, 3))
  val nd3x2x3 = NDArray(ar18, List(3, 2, 3))
  val nd3x3x2 = NDArray(ar18, List(3, 3, 2))
  val nd1x2x3x3 =
    NDArray((0 until 18).toArray.map(_.toDouble), List(1, 2, 3, 3))
  val nd1x2x2 = NDArray(mat.ones(2, 2).toArray, List(1, 2, 2))
  val nd1x2x2x2 = NDArray(mat.ones(2, 4).toArray, List(1, 2, 2, 2))
  val mat1x1 = Mat(Vec(1d))
  val mat3x2 = mat2x3.T
  val mat3x1 = Mat(Vec(1d, 2d, 3d))
  val mat3x1_2 = Mat(Vec(2d, 3d, 4d))
  def t2x3 = TensorHelpers.fromMat(mat2x3)
  val mat2x3_2 = Mat(Vec(-1d, 2d), Vec(3d, -4d), Vec(5d, 6d))
  def t3x2 = ATen.transpose(t2x3, 0, 1)

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
        Vec(fun(m, false, true)._1).roundTo(4) == Vec(expectedValue).roundTo(
          4
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
    Scope.leak { implicit scope =>
      val x1 = const(t2x3)
      val L = x1.sum
      assert(
        TensorHelpers
          .toMat(L.value) == Mat(Vec(TensorHelpers.toMat(t2x3).toVec.sum2))
      )
      L.backprop()
      assert(x1.partialDerivative.isEmpty)
      ()
    }
  }
  test("param is accumulating gradients") {
    Scope.leak { implicit scope =>
      val x1 = param(t2x3)
      val L = x1.sum
      // assert(L.value == Mat(Vec(mat2x3.toVec.sum2)))
      L.backprop()
      assert(x1.partialDerivative.get.toMat == mat.ones(2, 3))
      ()
    }
  }

  testGradientAndValue("sum")(mat2x3, 21d) { (m, doBackprop, cuda) =>
    Scope.leak { implicit scope =>
      val x1 = param(TensorHelpers.fromMat(m, cuda))
      val L = x1.sum
      if (doBackprop) {
        L.backprop()
      }
      (
        TensorHelpers.toMat(L.value).raw(0),
        x1.partialDerivative.map(t => t.toMat)
      )
    }
  }
  testGradientAndValue("colSum")(mat2x3, 21d) { (m, doBackprop, cuda) =>
    Scope.leak { implicit scope =>
      val x1 = param(TensorHelpers.fromMat(m, cuda))
      val L = x1.colSum.sum
      if (doBackprop) {
        L.backprop()
      }
      (
        TensorHelpers.toMat(L.value).raw(0),
        x1.partialDerivative.map(t => t.toMat)
      )
    }
  }
  testGradientAndValue("rowSum")(mat2x3, 21d) { (m, doBackprop, cuda) =>
    Scope.leak { implicit scope =>
      val x1 = param(TensorHelpers.fromMat(m, cuda))
      val L = x1.rowSum.sum
      if (doBackprop) {
        L.backprop()
      }
      (
        TensorHelpers.toMat(L.value).raw(0),
        x1.partialDerivative.map(t => t.toMat)
      )
    }
  }
  testGradientAndValue("assign - right")(mat2x3, 21d) { (m, doBackprop, cuda) =>
    Scope.leak { implicit scope =>
      val x1 = param(TensorHelpers.fromMat(m, cuda))
      val x2 = param(TensorHelpers.fromMat(mat2x3 * 2, cuda))
      val L = x2.assign(x1).sum
      if (doBackprop) {
        L.backprop()
      }
      (
        TensorHelpers.toMat(L.value).raw(0),
        x1.partialDerivative.map(t => t.toMat)
      )
    }
  }
  testGradientAndValue("assign - left")(mat2x3, 42d) { (m, doBackprop, cuda) =>
    Scope.leak { implicit scope =>
      val x2 = param(TensorHelpers.fromMat(m, cuda))
      val x1 = param(TensorHelpers.fromMat(mat2x3 * 2, cuda))
      val L = x2.assign(x1).sum
      if (doBackprop) {
        L.backprop()
      }
      (
        TensorHelpers.toMat(L.value).raw(0),
        x2.partialDerivative.map(t => t.toMat)
      )
    }
  }
  testGradientAndValue("add broadcasted - left")(Mat(Vec(1d)), 48d) {
    (m, doBackprop, cuda) =>
      Scope.leak { implicit scope =>
        val x1 = param(TensorHelpers.fromMat(m, cuda))
        val x2 = param(TensorHelpers.fromMat(mat2x3 * 2, cuda))
        val L = (x1.+(x2)).sum
        if (doBackprop) {
          L.backprop()
        }
        (
          TensorHelpers.toMat(L.value).raw(0),
          x1.partialDerivative.map(t => t.toMat)
        )
      }
  }
  testGradientAndValue("add - left")(mat2x3, 63d) { (m, doBackprop, cuda) =>
    Scope.leak { implicit scope =>
      val x1 = param(TensorHelpers.fromMat(m, cuda))
      val x2 = param(TensorHelpers.fromMat(mat2x3 * 2, cuda))
      val L = x1.+(x2).sum
      if (doBackprop) {
        L.backprop()
      }
      (
        TensorHelpers.toMat(L.value).raw(0),
        x1.partialDerivative.map(t => t.toMat)
      )
    }
  }
  testGradientAndValue("add broadcasted - right")(Mat(Vec(1d)), 48d) {
    (m, doBackprop, cuda) =>
      Scope.leak { implicit scope =>
        val x1 = param(TensorHelpers.fromMat(m, cuda))
        val x2 = param(TensorHelpers.fromMat(mat2x3 * 2, cuda))
        val L = (x2.+(x1)).sum
        if (doBackprop) {
          L.backprop()
        }
        (
          TensorHelpers.toMat(L.value).raw(0),
          x1.partialDerivative.map(t => t.toMat)
        )
      }
  }
  testGradientAndValue("add - right")(mat2x3, 63d) { (m, doBackprop, cuda) =>
    Scope.leak { implicit scope =>
      val x1 = param(TensorHelpers.fromMat(m, cuda))
      val x2 = param(TensorHelpers.fromMat(mat2x3 * 2, cuda))
      val L = x2.+(x1).sum
      if (doBackprop) {
        L.backprop()
      }
      (
        TensorHelpers.toMat(L.value).raw(0),
        x1.partialDerivative.map(t => t.toMat)
      )
    }
  }
  testGradientAndValue("minus - left")(mat2x3, -21d) { (m, doBackprop, cuda) =>
    Scope.leak { implicit scope =>
      val x1 = param(TensorHelpers.fromMat(m, cuda))
      val x2 = param(TensorHelpers.fromMat(mat2x3 * 2, cuda))
      val L = x1.-(x2).sum
      if (doBackprop) {
        L.backprop()
      }
      (
        TensorHelpers.toMat(L.value).raw(0),
        x1.partialDerivative.map(t => t.toMat)
      )
    }
  }
  testGradientAndValue("minus broadcasted - left")(mat1x1, -36d) {
    (m, doBackprop, cuda) =>
      Scope.leak { implicit scope =>
        val x1 = param(TensorHelpers.fromMat(m, cuda))
        val x2 = param(TensorHelpers.fromMat(mat2x3 * 2, cuda))
        val L = x1.-(x2).sum
        if (doBackprop) {
          L.backprop()
        }
        (
          TensorHelpers.toMat(L.value).raw(0),
          x1.partialDerivative.map(t => t.toMat)
        )
      }
  }
  testGradientAndValue("minus broadcasted - right")(mat1x1, 36d) {
    (m, doBackprop, cuda) =>
      Scope.leak { implicit scope =>
        val x1 = param(TensorHelpers.fromMat(m, cuda))
        val x2 = param(TensorHelpers.fromMat(mat2x3 * 2, cuda))
        val L = x2.-(x1).sum
        if (doBackprop) {
          L.backprop()
        }
        (
          TensorHelpers.toMat(L.value).raw(0),
          x1.partialDerivative.map(t => t.toMat)
        )
      }
  }
  testGradientAndValue("minus - right")(mat2x3, 21d) { (m, doBackprop, cuda) =>
    Scope.leak { implicit scope =>
      val x1 = param(TensorHelpers.fromMat(m, cuda))
      val x2 = param(TensorHelpers.fromMat(mat2x3 * 2, cuda))
      val L = x2.-(x1).sum
      if (doBackprop) {
        L.backprop()
      }
      (
        TensorHelpers.toMat(L.value).raw(0),
        x1.partialDerivative.map(t => t.toMat)
      )
    }
  }
  testGradientAndValue("constmult")(mat2x3, 42d) { (m, doBackprop, cuda) =>
    Scope.leak { implicit scope =>
      val x1 = param(TensorHelpers.fromMat(m, cuda))
      val L = x1.*(2d).sum
      if (doBackprop) {
        L.backprop()
      }
      (
        TensorHelpers.toMat(L.value).raw(0),
        x1.partialDerivative.map(t => t.toMat)
      )
    }
  }
  testGradientAndValue("constadd")(mat2x3, 33d) { (m, doBackprop, cuda) =>
    Scope.leak { implicit scope =>
      val x1 = param(TensorHelpers.fromMat(m, cuda))
      val L = x1.+(2d).sum
      if (doBackprop) {
        L.backprop()
      }
      (
        TensorHelpers.toMat(L.value).raw(0),
        x1.partialDerivative.map(t => t.toMat)
      )
    }
  }
  testGradientAndValue("mult broadcasted - left")(mat1x1, 42d) {
    (m, doBackprop, cuda) =>
      Scope.leak { implicit scope =>
        val x1 = param(TensorHelpers.fromMat(m, cuda))
        val x2 = param(TensorHelpers.fromMat(mat2x3 * 2, cuda))
        val L = x1.*(x2).sum
        if (doBackprop) {
          L.backprop()
        }
        (
          TensorHelpers.toMat(L.value).raw(0),
          x1.partialDerivative.map(t => t.toMat)
        )
      }
  }
  testGradientAndValue("mult broadcasted - right")(mat1x1, 42d) {
    (m, doBackprop, cuda) =>
      Scope.leak { implicit scope =>
        val x1 = param(TensorHelpers.fromMat(m, cuda))
        val x2 = param(TensorHelpers.fromMat(mat2x3 * 2, cuda))
        val L = x2.*(x1).sum
        if (doBackprop) {
          L.backprop()
        }
        (
          TensorHelpers.toMat(L.value).raw(0),
          x1.partialDerivative.map(t => t.toMat)
        )
      }
  }
  testGradientAndValue("mult - right")(mat2x3, 182d) { (m, doBackprop, cuda) =>
    Scope.leak { implicit scope =>
      val x1 = param(TensorHelpers.fromMat(m, cuda))
      val x2 = param(TensorHelpers.fromMat(mat2x3 * 2, cuda))
      val L = x2.*(x1).sum
      if (doBackprop) {
        L.backprop()
      }
      (
        TensorHelpers.toMat(L.value).raw(0),
        x1.partialDerivative.map(t => t.toMat)
      )
    }
  }

  testGradientAndValue("div - left")(mat2x3, 3d) { (m, doBackprop, cuda) =>
    Scope.leak { implicit scope =>
      val x1 = param(TensorHelpers.fromMat(m, cuda))
      val x2 = param(TensorHelpers.fromMat(mat2x3 * 2, cuda))
      val L = x1./(x2).sum
      if (doBackprop) {
        L.backprop()
      }
      (
        TensorHelpers.toMat(L.value).raw(0),
        x1.partialDerivative.map(t => t.toMat)
      )
    }
  }
  testGradientAndValue("div broadcasted - left")(mat1x1, 1.225d) {
    (m, doBackprop, cuda) =>
      Scope.leak { implicit scope =>
        val x1 = param(TensorHelpers.fromMat(m, cuda))
        val x2 = param(TensorHelpers.fromMat(mat2x3 * 2, cuda))
        val L = x1./(x2).sum
        if (doBackprop) {
          L.backprop()
        }
        (
          TensorHelpers.toMat(L.value).raw(0),
          x1.partialDerivative.map(t => t.toMat)
        )
      }
  }
  testGradientAndValue("div - right")(mat2x3, 12d) { (m, doBackprop, cuda) =>
    Scope.leak { implicit scope =>
      val x1 = param(TensorHelpers.fromMat(m, cuda))
      val x2 = param(TensorHelpers.fromMat(mat2x3 * 2, cuda))
      val L = x2./(x1).sum
      if (doBackprop) {
        L.backprop()
      }
      (
        TensorHelpers.toMat(L.value).raw(0),
        x1.partialDerivative.map(t => t.toMat)
      )
    }
  }

  testGradientAndValue("mm - left")(mat2x3, 358d) { (m, doBackprop, cuda) =>
    Scope.leak { implicit scope =>
      val x1 = param(TensorHelpers.fromMat(m, cuda))
      val x2 = param(TensorHelpers.fromMat(mat3x2 * 2, cuda))
      val L = x1.mm(x2).sum
      if (doBackprop) {
        L.backprop()
      }
      (
        TensorHelpers.toMat(L.value).raw(0),
        x1.partialDerivative.map(t => t.toMat)
      )
    }
  }
  testGradientAndValue("sparse mm - right")(mat2x3, 54d) {
    (m, doBackprop, cuda) =>
      Scope.leak { implicit scope =>
        val x1 = param(TensorHelpers.fromMat(m, cuda))
        val x2 = {
          val idx =
            TensorHelpers.fromLongMat(Mat(Vec(1L, 0L), Vec(0L, 1L)), cuda)
          val values = TensorHelpers.fromVec(Vec(2d, 3d), cuda)
          val topt = if (cuda) TensorOptions.d.cuda() else TensorOptions.d
          val sp = ATen.sparse_coo_tensor(idx, values, Array(3, 2), topt)
          const(sp)
        }
        val L = x2.mm(x1).sum
        if (doBackprop) {
          L.backprop()
        }
        (
          TensorHelpers.toMat(L.value).raw(0),
          x1.partialDerivative.map(t => t.toMat)
        )
      }
  }
  testGradientAndValue("mm - right")(mat2x3, 450d) { (m, doBackprop, cuda) =>
    Scope.leak { implicit scope =>
      val x1 = param(TensorHelpers.fromMat(m, cuda))
      val x2 = param(TensorHelpers.fromMat(mat3x2 * 2, cuda))
      val L = x2.mm(x1).sum
      if (doBackprop) {
        L.backprop()
      }
      (
        TensorHelpers.toMat(L.value).raw(0),
        x1.partialDerivative.map(t => t.toMat)
      )
    }
  }
  testGradientAndValue("crossentropy - left")(mat2x3, -182.0) {
    (m, doBackprop, cuda) =>
      Scope.leak { implicit scope =>
        val x1 = param(TensorHelpers.fromMat(m, cuda))
        val x2 = param(TensorHelpers.fromMat(mat2x3 * 2, cuda))

        val L = x1.crossEntropy(x2).sum
        if (doBackprop) {
          L.backprop()
        }
        (
          TensorHelpers.toMat(L.value).raw(0),
          x1.partialDerivative.map(t => t.toMat)
        )
      }
  }
  testGradientAndValue("crossentropy - right")(mat2x3, -182.0) {
    (m, doBackprop, cuda) =>
      Scope.leak { implicit scope =>
        val x1 = param(TensorHelpers.fromMat(m, cuda))
        val x2 = param(TensorHelpers.fromMat(mat2x3 * 2, cuda))
        val L = x2.crossEntropy(x1).sum
        if (doBackprop) {
          L.backprop()
        }
        (
          TensorHelpers.toMat(L.value).raw(0),
          x1.partialDerivative.map(t => t.toMat)
        )
      }
  }

  testGradientAndValue("relu")(mat2x3_2, 16d) { (m, doBackprop, cuda) =>
    Scope.leak { implicit scope =>
      val x1 = param(TensorHelpers.fromMat(m, cuda))
      val L = x1.relu.sum
      if (doBackprop) {
        L.backprop()
      }
      (
        TensorHelpers.toMat(L.value).raw(0),
        x1.partialDerivative.map(t => t.toMat)
      )
    }
  }
  testGradientAndValue("gelu")(mat2x3_2, 15.7917) { (m, doBackprop, cuda) =>
    Scope.leak { implicit scope =>
      val x1 = param(TensorHelpers.fromMat(m, cuda))
      val L = x1.gelu.sum
      if (doBackprop) {
        L.backprop()
      }
      (
        TensorHelpers.toMat(L.value).raw(0),
        x1.partialDerivative.map(t => t.toMat)
      )
    }
  }
  testGradientAndValue("sigmoid")(mat2x3_2, 4.1111) { (m, doBackprop, cuda) =>
    Scope.leak { implicit scope =>
      val x1 = param(TensorHelpers.fromMat(m, cuda))
      val L = x1.sigmoid.sum
      if (doBackprop) {
        L.backprop()
      }
      (
        TensorHelpers.toMat(L.value).raw(0),
        x1.partialDerivative.map(t => t.toMat)
      )
    }
  }
  testGradientAndValue("exp")(mat2x3_2, 579.7027406974902) {
    (m, doBackprop, cuda) =>
      Scope.leak { implicit scope =>
        val x1 = param(TensorHelpers.fromMat(m, cuda))
        val L = x1.exp.sum
        if (doBackprop) {
          L.backprop()
        }
        (
          TensorHelpers.toMat(L.value).raw(0),
          x1.partialDerivative.map(t => t.toMat)
        )
      }
  }
  testGradientAndValue("log")(mat2x3, 6.579251212010101) {
    (m, doBackprop, cuda) =>
      Scope.leak { implicit scope =>
        val x1 = param(TensorHelpers.fromMat(m, cuda))
        val L = x1.log.sum
        if (doBackprop) {
          L.backprop()
        }
        (
          TensorHelpers.toMat(L.value).raw(0),
          x1.partialDerivative.map(t => t.toMat)
        )
      }
  }
  testGradientAndValue("log1p")(mat2x3, 8.5252) { (m, doBackprop, cuda) =>
    Scope.leak { implicit scope =>
      val x1 = param(TensorHelpers.fromMat(m, cuda))
      val L = x1.log1p.sum
      if (doBackprop) {
        L.backprop()
      }
      (
        TensorHelpers.toMat(L.value).raw(0),
        x1.partialDerivative.map(t => t.toMat)
      )
    }
  }
  testGradientAndValue("sin")(mat2x3_2, -0.27259082747648367) {
    (m, doBackprop, cuda) =>
      Scope.leak { implicit scope =>
        val x1 = param(TensorHelpers.fromMat(m, cuda))
        val L = x1.sin.sum
        if (doBackprop) {
          L.backprop()
        }
        (
          TensorHelpers.toMat(L.value).raw(0),
          x1.partialDerivative.map(t => t.toMat)
        )
      }
  }
  testGradientAndValue("cos")(mat2x3_2, -0.2756481760294678) {
    (m, doBackprop, cuda) =>
      Scope.leak { implicit scope =>
        val x1 = param(TensorHelpers.fromMat(m, cuda))
        val L = x1.cos.sum
        if (doBackprop) {
          L.backprop()
        }
        (
          TensorHelpers.toMat(L.value).raw(0),
          x1.partialDerivative.map(t => t.toMat)
        )
      }
  }
  testGradientAndValue("tan")(mat2x3_2, -8.71433661097161) {
    (m, doBackprop, cuda) =>
      Scope.leak { implicit scope =>
        val x1 = param(TensorHelpers.fromMat(m, cuda))
        val L = x1.tan.sum
        if (doBackprop) {
          L.backprop()
        }
        (
          TensorHelpers.toMat(L.value).raw(0),
          x1.partialDerivative.map(t => t.toMat)
        )
      }
  }

  testGradientAndValue("atan")(mat2x3_2, 3.02402707945215) {
    (m, doBackprop, cuda) =>
      Scope.leak { implicit scope =>
        val x1 = param(TensorHelpers.fromMat(m, cuda))
        val L = x1.atan.sum
        if (doBackprop) {
          L.backprop()
        }
        (
          TensorHelpers.toMat(L.value).raw(0),
          x1.partialDerivative.map(t => t.toMat)
        )
      }
  }
  testGradientAndValue("capped exp")(mat3x1, 2.6065) { (m, doBackprop, cuda) =>
    Scope.leak { implicit scope =>
      val x1 = param(TensorHelpers.fromMat(m, cuda))
      val out = CappedShiftedNegativeExponential(scope, x1, 2.5d).value
      assert(
        out.toMat.roundTo(4) == Mat(Vec(1d, 1d, math.exp(-0.5))).roundTo(4)
      )
      val L = out.sum
      if (doBackprop) {
        L.backprop()
      }
      (
        TensorHelpers.toMat(L.value).raw(0),
        x1.partialDerivative.map(t => t.toMat)
      )
    }
  }
  testGradientAndValue("pow")(mat2x3_2, 91d) { (m, doBackprop, cuda) =>
    Scope.leak { implicit scope =>
      val x1 = param(TensorHelpers.fromMat(m, cuda))
      val L = x1.pow(2d).sum
      if (doBackprop) {
        L.backprop()
      }
      (
        TensorHelpers.toMat(L.value).raw(0),
        x1.partialDerivative.map(t => t.toMat)
      )
    }
  }
  testGradientAndValue("euclidean distance wrt a")(mat2x3_2, 10d) {
    (m, doBackprop, cuda) =>
      Scope.leak { implicit scope =>
        val x1 = param(TensorHelpers.fromMat(m, cuda))
        val out =
          x1.euclideanDistance(const(TensorHelpers.fromMat(mat2x3, cuda)), 1)
        assert(out.shape == List(2, 1))
        assert(out.toMat.roundTo(4) == Mat(Vec(2d, 8d)))
        val L = out.sum
        if (doBackprop) {
          L.backprop()
        }
        (
          TensorHelpers.toMat(L.value).raw(0),
          x1.partialDerivative.map(t => t.toMat)
        )
      }
  }
  testGradientAndValue("euclidean distance wrt b")(mat2x3_2, 10d) {
    (m, doBackprop, cuda) =>
      Scope.leak { implicit scope =>
        val x1 = param(TensorHelpers.fromMat(m, cuda))
        val out =
          const(TensorHelpers.fromMat(mat2x3, cuda)).euclideanDistance(x1, 1)
        assert(out.shape == List(2, 1))
        assert(out.toMat.roundTo(4) == Mat(Vec(2d, 8d)))
        val L = out.sum
        if (doBackprop) {
          L.backprop()
        }
        (
          TensorHelpers.toMat(L.value).raw(0),
          x1.partialDerivative.map(t => t.toMat)
        )
      }
  }
  testGradientAndValue("pow  2")(mat1x1, 11d) { (m, doBackprop, cuda) =>
    Scope.leak { implicit scope =>
      val x1 = param(TensorHelpers.fromMat(m, cuda))
      val x2 = param(TensorHelpers.fromMat(mat2x3_2, cuda))
      val L = x2.pow(x1).sum
      if (doBackprop) {
        L.backprop()
      }
      (
        TensorHelpers.toMat(L.value).raw(0),
        x1.partialDerivative.map(t => t.toMat)
      )
    }
  }
  testGradientAndValue("tanh")(mat2x3_2, 2.1981) { (m, doBackprop, cuda) =>
    Scope.leak { implicit scope =>
      val x1 = param(TensorHelpers.fromMat(m, cuda))
      val L = x1.tanh.sum
      if (doBackprop) {
        L.backprop()
      }
      (
        TensorHelpers.toMat(L.value).raw(0),
        x1.partialDerivative.map(t => t.toMat)
      )
    }
  }
  testGradientAndValue("softmax")(mat2x3_2, -22.441910257332836) {
    (m, doBackprop, cuda) =>
      Scope.leak { implicit scope =>
        val x1 = param(TensorHelpers.fromMat(m, cuda))
        val L = x1.logSoftMax(dim = 1).sum
        if (doBackprop) {
          L.backprop()
        }
        (
          TensorHelpers.toMat(L.value).raw(0),
          x1.partialDerivative.map(t => t.toMat)
        )
      }
  }
  testGradientAndValue("squaredFrobenius")(mat2x3_2, 91d) {
    (m, doBackprop, cuda) =>
      Scope.leak { implicit scope =>
        val x1 = param(TensorHelpers.fromMat(m, cuda))
        val L = x1.squaredFrobenius.sum
        if (doBackprop) {
          L.backprop()
        }
        (
          TensorHelpers.toMat(L.value).raw(0),
          x1.partialDerivative.map(t => t.toMat)
        )
      }
  }
  testGradientAndValue("transpose")(mat2x3_2, 11d) { (m, doBackprop, cuda) =>
    Scope.leak { implicit scope =>
      val x1 = param(TensorHelpers.fromMat(m, cuda))
      val L = x1.t.sum
      if (doBackprop) {
        L.backprop()
      }
      (
        TensorHelpers.toMat(L.value).raw(0),
        x1.partialDerivative.map(t => t.toMat)
      )
    }
  }
  testGradientAndValue("mean")(mat2x3_2, 5.5d) { (m, doBackprop, cuda) =>
    Scope.leak { implicit scope =>
      val x1 = param(TensorHelpers.fromMat(m, cuda))
      val L = x1.mean(List(0)).sum
      if (doBackprop) {
        L.backprop()
      }
      (
        TensorHelpers.toMat(L.value).raw(0),
        x1.partialDerivative.map(t => t.toMat)
      )
    }
  }
  testGradientAndValue("mse loss")(mat3x1, 1d) { (m, doBackprop, cuda) =>
    Scope.leak { implicit scope =>
      val x1 = param(TensorHelpers.fromMat(m, cuda))
      val x2 = TensorHelpers.fromMat(mat3x1_2, cuda)
      val L = x1.mseLoss(ATen.squeeze_0(x2)).sum
      if (doBackprop) {
        L.backprop()
      }
      (
        TensorHelpers.toMat(L.value).raw(0),
        x1.partialDerivative.map(t => t.toMat)
      )
    }
  }
  testGradientAndValue("l1 loss")(mat3x1, 1d) { (m, doBackprop, cuda) =>
    Scope.leak { implicit scope =>
      val x1 = param(TensorHelpers.fromMat(m, cuda))
      val x2 = TensorHelpers.fromMat(mat3x1_2, cuda)
      val L = x1.l1Loss(ATen.squeeze_0(x2)).sum
      if (doBackprop) {
        L.backprop()
      }
      (
        TensorHelpers.toMat(L.value).raw(0),
        x1.partialDerivative.map(t => t.toMat)
      )
    }
  }
  testGradientAndValue("l2 logistic regression loss")(
    mat2x3_2,
    151.0000008318073
  ) { (m, doBackprop, _) =>
    Scope.leak { implicit scope =>
      val w = param(TensorHelpers.fromMat(m))
      val data = const(TensorHelpers.fromMat(mat3x2))
      val y = const(TensorHelpers.fromMat(mat.ident(3)))
      val L =
        ((data
          .mm(w))
          .logSoftMax(dim = 1)
          .crossEntropy(y)
          .sum + w.squaredFrobenius)
      if (doBackprop) {
        L.backprop()
      }
      (
        TensorHelpers.toMat(L.value).raw(0),
        w.partialDerivative.map(t => t.toMat)
      )
    }
  }
  testGradientAndValue("l2 logistic regression loss - nll_loss")(
    mat2x3_2,
    151.0000008318073
  ) { (m, doBackprop, _) =>
    Scope.leak { implicit scope =>
      val w = param(TensorHelpers.fromMat(m))
      val data = const(TensorHelpers.fromMat(mat3x2))
      val y =
        const(ATen.squeeze_0(TensorHelpers.fromLongMat(Mat(Vec(0L, 1L, 2L)))))
      val classWeights = ATen.ones(Array(3), w.value.options())
      val L =
        ((data
          .mm(w))
          .logSoftMax(dim = 1)
          .nllLoss(y.value, 3, classWeights, Sum) + w.squaredFrobenius)
      if (doBackprop) {
        L.backprop()
      }
      (
        TensorHelpers.toMat(L.value).raw(0),
        w.partialDerivative.map(t => t.toMat)
      )
    }
  }
  testGradientAndValue("l2 logistic regression loss - nll_loss unreduced")(
    mat2x3_2,
    151.0000008318073
  ) { (m, doBackprop, _) =>
    Scope.leak { implicit scope =>
      val w = param(TensorHelpers.fromMat(m))
      val data = const(TensorHelpers.fromMat(mat3x2))
      val y =
        const(ATen.squeeze_0(TensorHelpers.fromLongMat(Mat(Vec(0L, 1L, 2L)))))
      val classWeights = ATen.ones(Array(3), w.value.options())
      val L =
        ((data
          .mm(w))
          .logSoftMax(dim = 1)
          .nllLoss(y.value, 3, classWeights, NoReduction)
          .sum + w.squaredFrobenius)
      if (doBackprop) {
        L.backprop()
      }
      (
        TensorHelpers.toMat(L.value).raw(0),
        w.partialDerivative.map(t => t.toMat)
      )
    }
  }
  testGradientAndValue("weight norm - wrt g")(mat2x3.row(Array(0)), 12.7279) {
    (m, doBackprop, cuda) =>
      Scope.leak { implicit scope =>
        val v = param(TensorHelpers.fromMat(mat.ones(2, 3), cuda))
        val g = param(TensorHelpers.fromMat(m, cuda))
        val L = WeightNorm(scope, v, g, 0).value.sum
        if (doBackprop) {
          L.backprop()
        }
        (
          TensorHelpers.toMat(L.value).raw(0),
          g.partialDerivative.map(t => t.toMat)
        )
      }
  }
  testGradientAndValue("weight norm - wrt v")(mat2x3, 4.1500) {
    (m, doBackprop, cuda) =>
      Scope.leak { implicit scope =>
        val v = param(TensorHelpers.fromMat(m, cuda))
        val g = param(TensorHelpers.fromMat(mat.ones(1, 3), cuda))
        val L = WeightNorm(scope, v, g, 0).value.sum
        if (doBackprop) {
          L.backprop()
        }
        (
          TensorHelpers.toMat(L.value).raw(0),
          v.partialDerivative.map(t => t.toMat)
        )
      }
  }
  testGradientAndValueND("mask")(nd1x2x2, 5d) { (m, doBackprop, cuda) =>
    Scope.leak { implicit scope =>
      val input =
        param(NDArray.tensorFromNDArray(m, cuda))
      val mask = {
        val q = NDArray.tensorFromNDArray(
          NDArray(Array(1d, 0d, 0d, 0d), List(1, 2, 2)),
          cuda
        )
        val sc = Tensor.scalarDouble(1d, q.options())
        param(ATen.eq_1(q, sc))
      }

      val output = MaskFill(scope, input, mask, 2d).value

      val L = output.sum
      if (doBackprop) {
        L.backprop()
      }
      (
        TensorHelpers.toMat(L.value).raw(0),
        input.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
      )
    }
  }
  testGradientAndValueND("index_fill")(nd1x2x2, 6d) { (m, doBackprop, cuda) =>
    Scope.leak { implicit scope =>
      val input =
        param(NDArray.tensorFromNDArray(m, cuda))
      val index =
        param(NDArray.tensorFromLongNDArray(NDArray(Array(1L), List(1)), cuda))

      val output = IndexFill(scope, input, 1L, index, 2d).value

      val L = output.sum
      if (doBackprop) {
        L.backprop()
      }
      (
        TensorHelpers.toMat(L.value).raw(0),
        input.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
      )
    }
  }
  testGradientAndValueND("expand as")(nd1x2x2, 16d) { (m, doBackprop, cuda) =>
    Scope.leak { implicit scope =>
      val input =
        param(NDArray.tensorFromNDArray(m, cuda))
      val other =
        NDArray.tensorFromNDArray(
          NDArray(
            org.saddle.vec.zeros(16).toArray,
            List(2, 2, 2, 2)
          ),
          cuda
        )
      val output = input.expandAs(other)
      assert(output.shape == other.shape)
      val L = output.sum
      if (doBackprop) {
        L.backprop()
      }
      (
        TensorHelpers.toMat(L.value).raw(0),
        input.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
      )
    }
  }
  testGradientAndValue("scatter sum")(mat2x3, 21d) { (m, doBackprop, cuda) =>
    Scope.leak { implicit scope =>
      val input =
        param(TensorHelpers.fromMat(m, cuda))
      val index =
        param(
          NDArray.tensorFromLongNDArray(
            NDArray(
              Array(0L, 0L, 1L, 0L, 1L, 1L),
              List(2, 3)
            ),
            cuda
          )
        )
      val output = input.scatterAdd(index, 0, 2)
      val L = output.sum
      if (doBackprop) {
        L.backprop()
      }
      (
        TensorHelpers.toMat(L.value).raw(0),
        input.partialDerivative.map(t => t.toMat)
      )
    }
  }

  testGradientAndValue("variance")(mat2x3, 8d) { (m, doBackprop, cuda) =>
    Scope.leak { implicit scope =>
      val input =
        param(TensorHelpers.fromMat(m, cuda))

      val output = input.variance(List(1))
      val L = output.sum
      if (doBackprop) {
        L.backprop()
      }
      (
        TensorHelpers.toMat(L.value).raw(0),
        input.partialDerivative.map(t => t.toMat)
      )
    }
  }
  testGradientAndValue("index sum")(mat2x3, 21d) { (m, doBackprop, cuda) =>
    Scope.leak { implicit scope =>
      val input =
        param(TensorHelpers.fromMat(m, cuda))
      val index =
        param(
          NDArray.tensorFromLongNDArray(
            NDArray(
              Array(1L, 1L),
              List(2)
            ),
            cuda
          )
        )
      val output = input.indexAdd(index, 0, 2)
      assert(
        output.toMat.roundTo(4) == Mat(Vec(0d, 0d, 0d), Vec(3d, 7d, 11d)).T
      )
      val L = output.sum
      if (doBackprop) {
        L.backprop()
      }
      (
        TensorHelpers.toMat(L.value).raw(0),
        input.partialDerivative.map(t => t.toMat)
      )
    }
  }
  testGradientAndValueND("index_select")(nd1x2x2, 6d) { (m, doBackprop, cuda) =>
    Scope.leak { implicit scope =>
      val input =
        param(NDArray.tensorFromNDArray(m, cuda))
      val index =
        param(
          NDArray
            .tensorFromLongNDArray(NDArray(Array(1L, 1L, 1L), List(3)), cuda)
        )

      val output = IndexSelect(scope, input, 1L, index).value

      val L = output.sum
      if (doBackprop) {
        L.backprop()
      }
      (
        TensorHelpers.toMat(L.value).raw(0),
        input.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
      )
    }
  }
  testGradientAndValueND("conv1d - wrt weights")(nd1x2x2, 30d) {
    (m, doBackprop, cuda) =>
      Scope.leak { implicit scope =>
        val input =
          param(NDArray.tensorFromNDArray(nd1x2x3, cuda))
        val weight = param(NDArray.tensorFromNDArray(m, cuda))

        val bias = param(TensorHelpers.fromVec(vec.ones(1), cuda))
        val output =
          Conv1D(scope, input, weight, bias, 1, 0, 1, 1L).value

        val L = output.sum
        if (doBackprop) {
          L.backprop()
        }
        (
          TensorHelpers.toMat(L.value).raw(0),
          weight.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
        )
      }
  }
  testGradientAndValueND("conv1d - wrt input")(nd1x2x3, 30d) {
    (m, doBackprop, cuda) =>
      Scope.leak { implicit scope =>
        val input =
          param(NDArray.tensorFromNDArray(m, cuda))
        val weight = param(NDArray.tensorFromNDArray(nd1x2x2, cuda))

        val bias = param(TensorHelpers.fromVec(vec.ones(1), cuda))
        val output =
          Conv1D(scope, input, weight, bias, 1, 0, 1, 1L).value

        val L = output.sum
        if (doBackprop) {
          L.backprop()
        }
        (
          TensorHelpers.toMat(L.value).raw(0),
          input.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
        )
      }
  }
  testGradientAndValueND("conv1d - padded - wrt weights")(nd1x2x2, 46d) {
    (m, doBackprop, cuda) =>
      Scope.leak { implicit scope =>
        val input =
          param(NDArray.tensorFromNDArray(nd1x2x3, cuda))
        val weight = param(NDArray.tensorFromNDArray(m, cuda))

        val bias = param(TensorHelpers.fromVec(vec.ones(1), cuda))
        val output =
          Conv1D(scope, input, weight, bias, 1L, 1L, 1L, 1L).value

        val L = output.sum
        if (doBackprop) {
          L.backprop()
        }
        (
          TensorHelpers.toMat(L.value).raw(0),
          weight.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
        )
      }
  }
  testGradientAndValueND("conv1d -padded - wrt input")(nd1x2x3, 46d) {
    (m, doBackprop, cuda) =>
      Scope.leak { implicit scope =>
        val input =
          param(NDArray.tensorFromNDArray(m, cuda))
        val weight = param(NDArray.tensorFromNDArray(nd1x2x2, cuda))

        val bias = param(TensorHelpers.fromVec(vec.ones(1), cuda))
        val output =
          Conv1D(scope, input, weight, bias, 1, 1, 1, 1L).value

        val L = output.sum
        if (doBackprop) {
          L.backprop()
        }
        (
          TensorHelpers.toMat(L.value).raw(0),
          input.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
        )
      }
  }
  testGradientAndValueND("conv1d - stride-2 - wrt weights")(nd1x2x2, 23d) {
    (m, doBackprop, cuda) =>
      Scope.leak { implicit scope =>
        val input =
          param(NDArray.tensorFromNDArray(nd1x2x3, cuda))
        val weight = param(NDArray.tensorFromNDArray(m, cuda))

        val bias = param(TensorHelpers.fromVec(vec.ones(1), cuda))
        val output =
          Conv1D(scope, input, weight, bias, 2L, 1L, 1L, 1L).value

        val L = output.sum
        if (doBackprop) {
          L.backprop()
        }
        (
          TensorHelpers.toMat(L.value).raw(0),
          weight.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
        )
      }
  }
  testGradientAndValueND("conv1d -stride-2 - wrt input")(nd1x2x3, 23d) {
    (m, doBackprop, cuda) =>
      Scope.leak { implicit scope =>
        val input =
          param(NDArray.tensorFromNDArray(m, cuda))
        val weight = param(NDArray.tensorFromNDArray(nd1x2x2, cuda))

        val bias = param(TensorHelpers.fromVec(vec.ones(1), cuda))
        val output =
          Conv1D(scope, input, weight, bias, 2, 1, 1, 1L).value

        val L = output.sum
        if (doBackprop) {
          L.backprop()
        }
        (
          TensorHelpers.toMat(L.value).raw(0),
          input.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
        )
      }
  }
  testGradientAndValueND("conv1d -stride-2 - wrt bias")(ndx1, 23d) {
    (m, doBackprop, cuda) =>
      Scope.leak { implicit scope =>
        val input =
          param(NDArray.tensorFromNDArray(nd1x2x3, cuda))
        val weight = param(NDArray.tensorFromNDArray(nd1x2x2, cuda))

        val bias = param(NDArray.tensorFromNDArray(m, cuda))
        val output =
          Conv1D(scope, input, weight, bias, 2, 1, 1, 1L).value

        val L = output.sum
        if (doBackprop) {
          L.backprop()
        }
        (
          TensorHelpers.toMat(L.value).raw(0),
          bias.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
        )
      }
  }
  testGradientAndValueND("conv2d - wrt weights")(nd1x2x2x2, 276d) {
    (m, doBackprop, cuda) =>
      Scope.leak { implicit scope =>
        val input =
          param(NDArray.tensorFromNDArray(nd1x2x3x3, cuda))
        val weight = param(NDArray.tensorFromNDArray(m, cuda))

        val bias = param(TensorHelpers.fromVec(vec.ones(1), cuda))
        val output =
          Conv2D(scope, input, weight, bias, 1, 0, 1, 1L).value

        val L = output.sum
        if (doBackprop) {
          L.backprop()
        }
        (
          TensorHelpers.toMat(L.value).raw(0),
          weight.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
        )
      }
  }
  testGradientAndValueND("conv2d - wrt input")(nd1x2x3x3, 276d) {
    (m, doBackprop, cuda) =>
      Scope.leak { implicit scope =>
        val input =
          param(NDArray.tensorFromNDArray(m, cuda))
        val weight = param(NDArray.tensorFromNDArray(nd1x2x2x2, cuda))

        val bias = param(TensorHelpers.fromVec(vec.ones(1), cuda))
        val output =
          Conv2D(scope, input, weight, bias, 1, 0, 1, 1L).value

        val L = output.sum
        if (doBackprop) {
          L.backprop()
        }
        (
          TensorHelpers.toMat(L.value).raw(0),
          input.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
        )
      }
  }
  testGradientAndValueND("conv2d - padded - wrt input")(nd1x2x3x3, 628d) {
    (m, doBackprop, cuda) =>
      Scope.leak { implicit scope =>
        val input =
          param(NDArray.tensorFromNDArray(m, cuda))
        val weight = param(NDArray.tensorFromNDArray(nd1x2x2x2, cuda))

        val bias = param(TensorHelpers.fromVec(vec.ones(1), cuda))
        val output =
          Conv2D(scope, input, weight, bias, 1, 1, 1, 1L).value

        val L = output.sum
        if (doBackprop) {
          L.backprop()
        }
        (
          TensorHelpers.toMat(L.value).raw(0),
          input.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
        )
      }
  }
  testGradientAndValueND("conv2d - wrt bias")(ndx1, 276d) {
    (m, doBackprop, cuda) =>
      Scope.leak { implicit scope =>
        val input =
          param(NDArray.tensorFromNDArray(nd1x2x3x3, cuda))
        val weight = param(NDArray.tensorFromNDArray(nd1x2x2x2, cuda))

        val bias = param(NDArray.tensorFromNDArray(m, cuda))
        val output =
          Conv2D(scope, input, weight, bias, 1, 0, 1, 1L).value

        val L = output.sum
        if (doBackprop) {
          L.backprop()
        }
        (
          TensorHelpers.toMat(L.value).raw(0),
          bias.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
        )
      }
  }

  testGradientAndValueND("maxpool1d padded")(nd1x2x3, 32d) {
    (m, doBackprop, cuda) =>
      Scope.leak { implicit scope =>
        val input =
          param(NDArray.tensorFromNDArray(m, cuda))
        val output =
          MaxPool1D(scope, input, 2, 1, 1, 1).value

        val L = output.sum
        if (doBackprop) {
          L.backprop()
        }
        (
          TensorHelpers.toMat(L.value).raw(0),
          input.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
        )
      }
  }
  testGradientAndValueND("maxpool1d unpadded")(nd1x2x3, 18d) {
    (m, doBackprop, cuda) =>
      Scope.leak { implicit scope =>
        val input =
          param(NDArray.tensorFromNDArray(m, cuda))

        val output =
          MaxPool1D(scope, input, 2, 1, 0, 1).value

        val L = output.sum
        if (doBackprop) {
          L.backprop()
        }
        (
          TensorHelpers.toMat(L.value).raw(0),
          input.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
        )
      }
  }
  testGradientAndValueND("maxpool1d strided")(nd1x2x3, 7d) {
    (m, doBackprop, cuda) =>
      Scope.leak { implicit scope =>
        val input =
          param(NDArray.tensorFromNDArray(m, cuda))

        val output =
          MaxPool1D(scope, input, 2, 2, 0, 1).value

        val L = output.sum
        if (doBackprop) {
          L.backprop()
        }
        (
          TensorHelpers.toMat(L.value).raw(0),
          input.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
        )
      }
  }
  testGradientAndValueND("maxpool2d strided")(nd1x2x3x3, 17d) {
    (m, doBackprop, cuda) =>
      Scope.leak { implicit scope =>
        val input =
          param(NDArray.tensorFromNDArray(m, cuda))

        val output =
          MaxPool2D(scope, input, 2, 2, 0, 1).value

        val L = output.sum
        if (doBackprop) {
          L.backprop()
        }
        (
          TensorHelpers.toMat(L.value).raw(0),
          input.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
        )
      }
  }
  testGradientAndValueND("maxpool2d strided padded")(nd1x2x3x3, 68d) {
    (m, doBackprop, cuda) =>
      Scope.leak { implicit scope =>
        val input =
          param(NDArray.tensorFromNDArray(m, cuda))

        val output =
          MaxPool2D(scope, input, 2, 2, 1, 1).value

        val L = output.sum
        if (doBackprop) {
          L.backprop()
        }
        (
          TensorHelpers.toMat(L.value).raw(0),
          input.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
        )
      }
  }
  testGradientAndValueND("avgpool2d strided padded")(nd1x2x3x3, 38.25) {
    (m, doBackprop, cuda) =>
      Scope.leak { implicit scope =>
        val input =
          param(NDArray.tensorFromNDArray(m, cuda))

        val output =
          AvgPool2D(scope, input, 2, 2, 1).value

        val L = output.sum
        if (doBackprop) {
          L.backprop()
        }
        (
          TensorHelpers.toMat(L.value).raw(0),
          input.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
        )
      }
  }
  testGradientAndValue("batch norm 1d - wrt to input")(mat2x3, 0d) {
    (m, doBackprop, cuda) =>
      Scope.leak { implicit scope =>
        val input =
          param(TensorHelpers.fromMat(m, cuda))
        val weight = param(TensorHelpers.fromVec(Vec(1d, 2d, 3d), cuda))

        val bias = param(TensorHelpers.fromVec(vec.zeros(3), cuda))
        val runningMean = TensorHelpers.fromVec(vec.ones(3), cuda)
        val runningVar = TensorHelpers.fromVec(vec.ones(3), cuda)

        val output =
          BatchNorm(
            scope,
            input,
            weight,
            bias,
            runningMean,
            runningVar,
            training = true,
            momentum = 0.1,
            eps = 1e-5
          ).value

        val L = output.sum
        if (doBackprop) {
          L.backprop()
        }
        (
          TensorHelpers.toMat(L.value).raw(0),
          input.partialDerivative.map(t => t.toMat)
        )
      }
  }
  testGradientAndValueND("batch norm 1d - wrt to weight")(ndx3, 0d) {
    (m, doBackprop, cuda) =>
      Scope.leak { implicit scope =>
        val input =
          param(TensorHelpers.fromMat(mat2x3, cuda))
        val weight = param(NDArray.tensorFromNDArray(m, cuda))

        val bias = param(TensorHelpers.fromVec(vec.zeros(3), cuda))
        val runningMean = TensorHelpers.fromVec(vec.ones(3), cuda)
        val runningVar = TensorHelpers.fromVec(vec.ones(3), cuda)

        val output =
          BatchNorm(
            scope,
            input,
            weight,
            bias,
            runningMean,
            runningVar,
            training = true,
            momentum = 0.1,
            eps = 1e-5
          ).value

        val L = output.sum
        if (doBackprop) {
          L.backprop()
        }
        (
          TensorHelpers.toMat(L.value).raw(0),
          weight.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
        )
      }
  }
  testGradientAndValueND("batch norm 1d - wrt to bias")(ndx3, 12d) {
    (m, doBackprop, cuda) =>
      Scope.leak { implicit scope =>
        val input =
          param(TensorHelpers.fromMat(mat2x3, cuda))
        val bias = param(NDArray.tensorFromNDArray(m, cuda))

        val weight = param(TensorHelpers.fromVec(vec.zeros(3), cuda))
        val runningMean = TensorHelpers.fromVec(vec.ones(3), cuda)
        val runningVar = TensorHelpers.fromVec(vec.ones(3), cuda)

        val output =
          BatchNorm(
            scope,
            input,
            weight,
            bias,
            runningMean,
            runningVar,
            training = true,
            momentum = 0.1,
            eps = 1e-5
          ).value

        val L = output.sum
        if (doBackprop) {
          L.backprop()
        }
        (
          TensorHelpers.toMat(L.value).raw(0),
          bias.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
        )
      }
  }
  testGradientAndValueND("bmm - wrt left")(nd3x2x3, 489d) {
    (m, doBackprop, cuda) =>
      Scope.leak { implicit scope =>
        val input =
          param(NDArray.tensorFromNDArray(m, cuda))
        val other = param(NDArray.tensorFromNDArray(nd3x3x2, cuda))
        val output = input.bmm(other)

        val L = output.sum
        if (doBackprop) {
          L.backprop()
        }
        (
          TensorHelpers.toMat(L.value).raw(0),
          input.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
        )
      }
  }
  testGradientAndValueND("bmm - wrt right")(nd3x3x2, 489d) {
    (m, doBackprop, cuda) =>
      Scope.leak { implicit scope =>
        val input =
          param(NDArray.tensorFromNDArray(m, cuda))
        val other = param(NDArray.tensorFromNDArray(nd3x2x3, cuda))
        val output = other.bmm(input)

        val L = output.sum
        if (doBackprop) {
          L.backprop()
        }
        (
          TensorHelpers.toMat(L.value).raw(0),
          input.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
        )
      }
  }
  testGradientAndValueND("batch norm 2d - wrt to input")(nd1x2x3, 6d) {
    (m, doBackprop, cuda) =>
      Scope.leak { implicit scope =>
        val input =
          param(NDArray.tensorFromNDArray(m, cuda))
        val bias = param(TensorHelpers.fromVec(vec.ones(6), cuda))

        val weight = param(TensorHelpers.fromVec(vec.ones(6), cuda))
        val runningMean = TensorHelpers.fromVec(vec.ones(6), cuda)
        val runningVar = TensorHelpers.fromVec(vec.ones(6), cuda)

        val output =
          BatchNorm(
            scope,
            input,
            weight,
            bias,
            runningMean,
            runningVar,
            training = true,
            momentum = 0.1,
            eps = 1e-5
          ).value

        val L = output.sum
        if (doBackprop) {
          L.backprop()
        }
        (
          TensorHelpers.toMat(L.value).raw(0),
          input.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
        )
      }
  }
  testGradientAndValueND("batch norm 2d - wrt to weights")(ndx6, 6d) {
    (m, doBackprop, cuda) =>
      Scope.leak { implicit scope =>
        val input =
          param(NDArray.tensorFromNDArray(nd1x2x3, cuda))
        val bias = param(TensorHelpers.fromVec(vec.ones(6), cuda))

        val weight = param(NDArray.tensorFromNDArray(m, cuda))
        val runningMean = TensorHelpers.fromVec(vec.ones(6), cuda)
        val runningVar = TensorHelpers.fromVec(vec.ones(6), cuda)

        val output =
          BatchNorm(
            scope,
            input,
            weight,
            bias,
            runningMean,
            runningVar,
            training = true,
            momentum = 0.1,
            eps = 1e-5
          ).value

        val L = output.sum
        if (doBackprop) {
          L.backprop()
        }
        (
          TensorHelpers.toMat(L.value).raw(0),
          weight.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
        )
      }
  }
  testGradientAndValueND("batch norm 2d - wrt to bias")(ndx6, 21d) {
    (m, doBackprop, cuda) =>
      Scope.leak { implicit scope =>
        val input =
          param(NDArray.tensorFromNDArray(nd1x2x3, cuda))
        val weight = param(TensorHelpers.fromVec(vec.ones(6), cuda))

        val bias = param(NDArray.tensorFromNDArray(m, cuda))
        val runningMean = TensorHelpers.fromVec(vec.ones(6), cuda)
        val runningVar = TensorHelpers.fromVec(vec.ones(6), cuda)

        val output =
          BatchNorm(
            scope,
            input,
            weight,
            bias,
            runningMean,
            runningVar,
            training = true,
            momentum = 0.1,
            eps = 1e-5
          ).value

        val L = output.sum
        if (doBackprop) {
          L.backprop()
        }
        (
          TensorHelpers.toMat(L.value).raw(0),
          bias.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
        )
      }
  }
  testGradientAndValueND("batch norm 3d - wrt to input")(nd1x2x3x3, 18d) {
    (m, doBackprop, cuda) =>
      Scope.leak { implicit scope =>
        val input =
          param(NDArray.tensorFromNDArray(m, cuda))
        val bias = param(TensorHelpers.fromVec(vec.ones(18), cuda))

        val weight = param(TensorHelpers.fromVec(vec.ones(18), cuda))
        val runningMean = TensorHelpers.fromVec(vec.ones(18), cuda)
        val runningVar = TensorHelpers.fromVec(vec.ones(18), cuda)

        val output =
          BatchNorm(
            scope,
            input,
            weight,
            bias,
            runningMean,
            runningVar,
            training = true,
            momentum = 0.1,
            eps = 1e-5
          ).value

        val L = output.sum
        if (doBackprop) {
          L.backprop()
        }
        (
          TensorHelpers.toMat(L.value).raw(0),
          input.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
        )
      }
  }
  testGradientAndValueND("batch norm 3d - wrt to weights")(ndx18, 18d) {
    (m, doBackprop, cuda) =>
      Scope.leak { implicit scope =>
        val input =
          param(NDArray.tensorFromNDArray(nd1x2x3x3, cuda))
        val bias = param(TensorHelpers.fromVec(vec.ones(18), cuda))

        val weight = param(NDArray.tensorFromNDArray(m, cuda))
        val runningMean = TensorHelpers.fromVec(vec.ones(18), cuda)
        val runningVar = TensorHelpers.fromVec(vec.ones(18), cuda)

        val output =
          BatchNorm(
            scope,
            input,
            weight,
            bias,
            runningMean,
            runningVar,
            training = true,
            momentum = 0.1,
            eps = 1e-5
          ).value

        val L = output.sum
        if (doBackprop) {
          L.backprop()
        }
        (
          TensorHelpers.toMat(L.value).raw(0),
          weight.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
        )
      }
  }
  testGradientAndValueND("batch norm 3d - wrt to bias")(ndx18, 63d) {
    (m, doBackprop, cuda) =>
      Scope.leak { implicit scope =>
        val input =
          param(NDArray.tensorFromNDArray(nd1x2x3x3, cuda))
        val weights = param(TensorHelpers.fromVec(vec.ones(18), cuda))

        val bias = param(NDArray.tensorFromNDArray(m, cuda))
        val runningMean = TensorHelpers.fromVec(vec.ones(18), cuda)
        val runningVar = TensorHelpers.fromVec(vec.ones(18), cuda)

        val output =
          BatchNorm(
            scope,
            input,
            weights,
            bias,
            runningMean,
            runningVar,
            training = true,
            momentum = 0.1,
            eps = 1e-5
          ).value

        val L = output.sum
        if (doBackprop) {
          L.backprop()
        }
        (
          TensorHelpers.toMat(L.value).raw(0),
          bias.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
        )
      }
  }
  testGradientAndValueND("BatchNorm2D - wrt to input")(nd1x2x3x3, 18d) {
    (m, doBackprop, cuda) =>
      Scope.leak { implicit scope =>
        val input =
          param(NDArray.tensorFromNDArray(m, cuda))
        val weights = param(TensorHelpers.fromVec(vec.ones(2), cuda))

        val bias = param(NDArray.tensorFromNDArray(ndx2, cuda))
        val runningMean = TensorHelpers.fromVec(vec.zeros(2), cuda)
        val runningVar = TensorHelpers.fromVec(vec.zeros(2), cuda)

        val output =
          BatchNorm2D(
            scope,
            input,
            weights,
            bias,
            runningMean,
            runningVar,
            training = true,
            momentum = 0.1,
            eps = 1e-5
          ).value

        val L = output.sum
        if (doBackprop) {
          L.backprop()
        }
        (
          TensorHelpers.toMat(L.value).raw(0),
          input.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
        )
      }
  }
  testGradientAndValueND("BatchNorm2D - wrt to weights")(ndx2, 18d) {
    (m, doBackprop, cuda) =>
      Scope.leak { implicit scope =>
        val input =
          param(NDArray.tensorFromNDArray(nd1x2x3x3, cuda))
        val weights = param(NDArray.tensorFromNDArray(m, cuda))

        val bias = param(NDArray.tensorFromNDArray(ndx2, cuda))
        val runningMean = TensorHelpers.fromVec(vec.zeros(2), cuda)
        val runningVar = TensorHelpers.fromVec(vec.zeros(2), cuda)

        val output =
          BatchNorm2D(
            scope,
            input,
            weights,
            bias,
            runningMean,
            runningVar,
            training = true,
            momentum = 0.1,
            eps = 1e-5
          ).value

        val L = output.sum
        if (doBackprop) {
          L.backprop()
        }
        (
          TensorHelpers.toMat(L.value).raw(0),
          weights.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
        )
      }
  }
  testGradientAndValueND("BatchNorm2D - wrt to bias")(ndx2, 18d) {
    (m, doBackprop, cuda) =>
      Scope.leak { implicit scope =>
        val input =
          param(NDArray.tensorFromNDArray(nd1x2x3x3, cuda))
        val bias = param(NDArray.tensorFromNDArray(m, cuda))

        val weights = param(NDArray.tensorFromNDArray(ndx2, cuda))
        val runningMean = TensorHelpers.fromVec(vec.zeros(2), cuda)
        val runningVar = TensorHelpers.fromVec(vec.zeros(2), cuda)

        val output =
          BatchNorm2D(
            scope,
            input,
            weights,
            bias,
            runningMean,
            runningVar,
            training = true,
            momentum = 0.1,
            eps = 1e-5
          ).value

        val L = output.sum
        if (doBackprop) {
          L.backprop()
        }
        (
          TensorHelpers.toMat(L.value).raw(0),
          bias.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
        )
      }
  }
  testGradientAndValueND("flatten ")(nd1x2x3x3, 153d) { (m, doBackprop, cuda) =>
    Scope.leak { implicit scope =>
      val input =
        param(NDArray.tensorFromNDArray(m, cuda))

      val output =
        FlattenLastDimensions(scope, input, 3).value

      assert(output.shape == List(1, 18))

      val L = output.sum
      if (doBackprop) {
        L.backprop()
      }
      (
        TensorHelpers.toMat(L.value).raw(0),
        input.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
      )
    }
  }
  testGradientAndValueND("select 0 0 ")(nd1x2x3x3, 153d) {
    (m, doBackprop, cuda) =>
      Scope.leak { implicit scope =>
        val input =
          param(NDArray.tensorFromNDArray(m, cuda))

        val output =
          input.select(0, 0)

        assert(output.shape == List(2, 3, 3))

        val L = output.sum
        if (doBackprop) {
          L.backprop()
        }
        (
          TensorHelpers.toMat(L.value).raw(0),
          input.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
        )
      }
  }
  testGradientAndValueND("select 2 1 ")(nd1x2x3x3, 51d) {
    (m, doBackprop, cuda) =>
      Scope.leak { implicit scope =>
        val input =
          param(NDArray.tensorFromNDArray(m, cuda))

        val output =
          input.select(2, 1)

        assert(output.shape == List(1, 2, 3))

        val L = output.sum
        if (doBackprop) {
          L.backprop()
        }
        (
          TensorHelpers.toMat(L.value).raw(0),
          input.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
        )
      }
  }
  testGradientAndValueND("cat 0 ")(nd1x2x3, 42d) { (m, doBackprop, cuda) =>
    Scope.leak { implicit scope =>
      val input =
        param(NDArray.tensorFromNDArray(m, cuda))

      val output =
        ConcatenateAddNewDim(scope, List(input, input)).value

      assert(output.shape == List(2, 1, 2, 3))

      val L = output.sum
      if (doBackprop) {
        L.backprop()
      }
      (
        TensorHelpers.toMat(L.value).raw(0),
        input.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
      )
    }
  }
  testGradientAndValueND("cat 1 ")(nd1x2x3, 84d) { (m, doBackprop, cuda) =>
    Scope.leak { implicit scope =>
      val input =
        param(NDArray.tensorFromNDArray(m, cuda))
      val t2 =
        param(NDArray.tensorFromNDArray(nd1x2x3_2, cuda))

      val output =
        Concatenate(scope, List(input, t2), 1).value

      assert(output.shape == List(1, 4, 3))

      val L = output.sum
      if (doBackprop) {
        L.backprop()
      }
      (
        TensorHelpers.toMat(L.value).raw(0),
        input.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
      )
    }
  }
  testGradientAndValueND("cat 2 ")(nd1x2x3, 84d) { (m, doBackprop, cuda) =>
    Scope.leak { implicit scope =>
      val input =
        param(NDArray.tensorFromNDArray(m, cuda))
      val t2 =
        param(NDArray.tensorFromNDArray(nd1x2x3_2, cuda))
      val output =
        Concatenate(scope, List(input, t2), 2).value

      assert(output.shape == List(1, 2, 6))

      val L = output.sum
      if (doBackprop) {
        L.backprop()
      }
      (
        TensorHelpers.toMat(L.value).raw(0),
        input.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
      )
    }
  }

  testGradientAndValueND("view 1 ")(nd1x2x3, 21d) { (m, doBackprop, cuda) =>
    Scope.leak { implicit scope =>
      val input =
        param(NDArray.tensorFromNDArray(m, cuda))

      val output = input.view(List(1, 1, 2, 3))

      assert(output.shape == List(1, 1, 2, 3))

      val L = output.sum
      if (doBackprop) {
        L.backprop()
      }
      (
        TensorHelpers.toMat(L.value).raw(0),
        input.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
      )
    }
  }
  testGradientAndValue("embedding ")(mat2x3, 240d) { (m, doBackprop, cuda) =>
    Scope.leak { implicit scope =>
      val weight =
        param(TensorHelpers.fromMat(m, cuda))
      val input =
        param(TensorHelpers.fromLongMat(mat.ones(4, 5).map(_.toLong), cuda))

      val output = Embedding(scope, input, weight).value

      assert(output.shape == List(4, 5, 3))

      val L = output.sum
      if (doBackprop) {
        L.backprop()
      }
      (
        TensorHelpers.toMat(L.value).raw(0),
        weight.partialDerivative.map(t => t.toMat)
      )
    }
  }
}
