package lamp.autograd

import org.saddle._
import org.saddle.linalg._
import org.saddle.ops.BinOps._
import org.scalatest.funsuite.AnyFunSuite
import lamp.nn.CudaTest
import lamp.Scope
import lamp.util.NDArray
import lamp.STen
import lamp.STenOptions
import lamp.saddle._

object NDArraySyntax {
  implicit class syntax[T](value: NDArray[T]) {
    def toVec(implicit st: ST[T]) = Vec(value.data)
  }
}

import NDArraySyntax._
class GradientSuite extends AnyFunSuite {
  val ar18 = Array(1d, 2d, 3d, 4d, 5d, 6d, 1d, 2d, 3d, 4d, 5d, 6d, 1d, 2d, 3d,
    4d, 5d, 6d)
  val mat2x3 = Mat(Vec(1d, 2d), Vec(3d, 4d), Vec(5d, 6d))
  val mat3x3 = Mat(Vec(1d, 2d, 0d), Vec(4d, 5d, 1d), Vec(6d, 7d, 0d)).innerM
  val mat2x2 = Mat(Vec(4d, 1d), Vec(6d, 2d)).T
  val mat2x2PD = Mat(Vec(4d, 1d), Vec(6d, 2d)).outerM
  val mat2x2PDCholeskyFactor = mat2x2PD.choleskyLower.get
  mat2x2PDCholeskyFactor.update(0, 1, 0d)

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
  val nd2x3x3 = NDArray(
    Array(10d, 2d, 0d, 4d, 5d, 1d, 6d, 7d, 0d, 1d, 2d, 0d, 4d, 5d, 1d, 6d, 7d,
      0d),
    List(2, 3, 3)
  )
  val nd1x3x3 = NDArray(
    Array(1d, 2d, 0d, 4d, 5d, 1d, 6d, 7d, 0d),
    List(1, 3, 3)
  )
  val nd1x2x3x3 =
    NDArray((0 until 18).toArray.map(_.toDouble), List(1, 2, 3, 3))
  val nd1x4x3x3 =
    NDArray((0 until 36).toArray.map(_.toDouble), List(1, 4, 3, 3))
  val nd1x2x2 = NDArray(mat.ones(2, 2).toArray, List(1, 2, 2))
  val nd1x2x2x2 = NDArray(mat.ones(2, 4).toArray, List(1, 2, 2, 2))
  val nd2x2x2x2 = NDArray(mat.ones(4, 4).toArray, List(2, 2, 2, 2))
  val mat1x1 = Mat(Vec(1d))
  val mat3x2 = mat2x3.T
  val mat3x1 = Mat(Vec(1d, 2d, 3d))
  val mat1x3 = Mat(Vec(1d, 2d, 3d)).T
  val mat3x1_2 = Mat(Vec(2d, 3d, 4d))
  val mat1x64 = Mat(vec.ones(64))
  def t2x3 = lamp.saddle.fromMat(mat2x3)(Scope.free)
  val mat2x3_2 = Mat(Vec(-1d, 2d), Vec(3d, -4d), Vec(5d, 6d))
  def t3x2 = t2x3.transpose(0, 1)(Scope.free)

  def diff(m: Mat[Double], eps: Double = 1e-6)(
      f: Mat[Double] => Double
  ): Mat[Double] = {
    mat.zeros(m.numRows, m.numCols).mapRows { case (row, i) =>
      (0 until row.length).map { j =>
        val epsM = mat.zeros(m.numRows, m.numCols)
        epsM(i, j) = eps
        (f(m + epsM) - f(m - epsM)) / (2 * eps)
      }.toVec
    }

  }
  private[lamp] def diffND(
      m: NDArray[Double]
  )(f: NDArray[Double] => Double): NDArray[Double] = {
    val eps = 1e-6

    NDArray.zeros(m.shape).mapWithIndex { case (_, idx) =>
      val epsM = NDArray.zeros(m.shape)
      epsM.set(idx, eps)
      val a = f(m + epsM)
      val b = f(m - epsM)
      val r = (a - b) / (2 * eps)
      r
    }

  }

  def testGradientAndValueCudaOnly(
      id: String
  )(m: Mat[Double], expectedValue: Double, eps: Double = 1e-6)(
      fun: (Mat[Double], Boolean) => (Double, Option[Mat[Double]])
  ) = {
    test(id + ": gradient is correct", CudaTest) {

      def diffNum(m: Mat[Double]) = diff(m, eps)(m => fun(m, false)._1)
      def diffAuto(m: Mat[Double]) = {
        fun(m, true)._2.get
      }
      assert(
        Vec(fun(m, false)._1).roundTo(4) == Vec(expectedValue).roundTo(
          4
        )
      )

      assert(diffAuto(m).roundTo(4) == diffNum(m).roundTo(4))
    }
  }
  def testGradientAndValue(
      id: String
  )(m: Mat[Double], expectedValue: Double, eps: Double = 1e-6)(
      fun: (Mat[Double], Boolean, Boolean) => (Double, Option[Mat[Double]])
  ) = {
    test(id + ": gradient is correct") {

      def diffNum(m: Mat[Double]) = diff(m, eps)(m => fun(m, false, false)._1)
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
  private[lamp] def testGradientAndValueND(
      id: String,
      cuda: Boolean = true,
      cpu: Boolean = true
  )(m: NDArray[Double], expectedValue: Double)(
      fun: (NDArray[Double], Boolean, Boolean) => (
          Double,
          Option[NDArray[Double]]
      )
  ) = {
    if (cpu) {
      test(id + ": gradient is correct") {

        def diffNum(m: NDArray[Double]) =
          diffND(m)(m => fun(m, false, false)._1)
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
    }
    if (cuda) {
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
  }

  test("constant is not accumulating gradients") {
    Scope.root { implicit scope =>
      val x1 = const(t2x3)
      val L = x1.sum
      assert(
        L.value.toMat == Mat(Vec(t2x3.toMat.toVec.sum2))
      )
      L.backprop()
      assert(x1.partialDerivative.isEmpty)
      ()
    }
  }
  test("param is accumulating gradients") {
    Scope.root { implicit scope =>
      val x1 = param(t2x3)
      val L = x1.sum
      // assert(L.value == Mat(Vec(mat2x3.toVec.sum2)))
      L.backprop()
      assert(x1.partialDerivative.get.toMat == mat.ones(2, 3))
      ()
    }
  }

  testGradientAndValueCudaOnly("scaled dot product attention - by q")(
    mat1x64,
    64d
  ) { (m, doBackprop) =>
    Scope.root { implicit scope =>
      val device = lamp.CudaDevice(0)
      val mSTen = device.to(lamp.saddle.fromMat(m).view(1, 8, 1, 8).castToFloat)
      val q = param(mSTen + 0.0)
      val k = param(device.to(STen.ones(List(1, 8, 1, 8), STenOptions.f)))
      val v = param(device.to(STen.ones(List(1, 8, 1, 8), STenOptions.f)))
      val r = new ScaledDotProductAttention(scope, q, k, v, false).value

      val L = r.sum

      if (doBackprop) {
        L.backprop()
      }
      (
        lamp.CPU.to(L.value).toMat.raw(0),
        q.partialDerivative.map(t => t.reshape(-1, 1).toMat)
      )
    }
  }
  testGradientAndValueCudaOnly("scaled dot product attention - by k")(
    mat1x64,
    64d
  ) { (m, doBackprop) =>
    Scope.root { implicit scope =>
      val device = lamp.CudaDevice(0)
      val mSTen = device.to(lamp.saddle.fromMat(m).view(1, 8, 1, 8).castToFloat)
      val q = param(device.to(STen.ones(List(1, 8, 1, 8), STenOptions.f)))
      val k = param(mSTen + 0.0)
      val v = param(device.to(STen.ones(List(1, 8, 1, 8), STenOptions.f)))
      val r = new ScaledDotProductAttention(scope, q, k, v, false).value

      val L = r.sum
      if (doBackprop) {
        L.backprop()
      }
      (
        lamp.CPU.to(L.value).toMat.raw(0),
        k.partialDerivative.map(t => t.reshape(-1, 1).toMat)
      )
    }
  }
  testGradientAndValueCudaOnly("scaled dot product attention - by v")(
    mat1x64,
    704d,
    eps = 1e-3
  ) { (m, doBackprop) =>
    Scope.root { implicit scope =>
      val device = lamp.CudaDevice(0)
      val mSTen = device.to(lamp.saddle.fromMat(m).view(1, 8, 1, 8).castToFloat)
      val q = param(device.to(STen.ones(List(1, 8, 1, 8), STenOptions.f)))
      val k = param(device.to(STen.ones(List(1, 8, 1, 8), STenOptions.f)))
      val v = param(mSTen + 10.0)
      val r = new ScaledDotProductAttention(scope, q, k, v, false).value

      val L = r.sum
      if (doBackprop) {
        L.backprop()
      }
      (
        lamp.CPU.to(L.value).toMat.raw(0),
        v.partialDerivative.map(t => t.reshape(-1, 1).toMat)
      )
    }
  }
  testGradientAndValue("sum")(mat2x3, 21d) { (m, doBackprop, cuda) =>
    Scope.root { implicit scope =>
      val x1 = param(lamp.saddle.fromMat(m, cuda))
      val L = x1.sum
      if (doBackprop) {
        L.backprop()
      }
      (
        L.value.toMat.raw(0),
        x1.partialDerivative.map(t => t.toMat)
      )
    }
  }
  testGradientAndValue("colSum")(mat2x3, 21d) { (m, doBackprop, cuda) =>
    Scope.root { implicit scope =>
      val x1 = param(lamp.saddle.fromMat(m, cuda))
      val L = x1.colSum.sum
      if (doBackprop) {
        L.backprop()
      }
      (
        L.value.toMat.raw(0),
        x1.partialDerivative.map(t => t.toMat)
      )
    }
  }
  testGradientAndValue("rowSum")(mat2x3, 21d) { (m, doBackprop, cuda) =>
    Scope.root { implicit scope =>
      val x1 = param(lamp.saddle.fromMat(m, cuda))
      val L = x1.rowSum.sum
      if (doBackprop) {
        L.backprop()
      }
      (
        L.value.toMat.raw(0),
        x1.partialDerivative.map(t => t.toMat)
      )
    }
  }
  testGradientAndValue("assign - right")(mat2x3, 21d) { (m, doBackprop, cuda) =>
    Scope.root { implicit scope =>
      val x1 = param(lamp.saddle.fromMat(m, cuda))
      val x2 = param(lamp.saddle.fromMat(mat2x3 * 2, cuda))
      val L = x2.assign(x1).sum
      if (doBackprop) {
        L.backprop()
      }
      (
        L.value.toMat.raw(0),
        x1.partialDerivative.map(t => t.toMat)
      )
    }
  }
  testGradientAndValue("assign - left")(mat2x3, 42d) { (m, doBackprop, cuda) =>
    Scope.root { implicit scope =>
      val x2 = param(lamp.saddle.fromMat(m, cuda))
      val x1 = param(lamp.saddle.fromMat(mat2x3 * 2, cuda))
      val L = x2.assign(x1).sum
      if (doBackprop) {
        L.backprop()
      }
      (
        L.value.toMat.raw(0),
        x2.partialDerivative.map(t => t.toMat)
      )
    }
  }
  testGradientAndValue("add broadcasted - left")(Mat(Vec(1d)), 48d) {
    (m, doBackprop, cuda) =>
      Scope.root { implicit scope =>
        val x1 = param(lamp.saddle.fromMat(m, cuda))
        val x2 = param(lamp.saddle.fromMat(mat2x3 * 2, cuda))
        val L = (x1.+(x2)).sum
        if (doBackprop) {
          L.backprop()
        }
        (
          L.value.toMat.raw(0),
          x1.partialDerivative.map(t => t.toMat)
        )
      }
  }
  testGradientAndValue("add - left")(mat2x3, 63d) { (m, doBackprop, cuda) =>
    Scope.root { implicit scope =>
      val x1 = param(lamp.saddle.fromMat(m, cuda))
      val x2 = param(lamp.saddle.fromMat(mat2x3 * 2, cuda))
      val L = x1.+(x2).sum
      if (doBackprop) {
        L.backprop()
      }
      (
        L.value.toMat.raw(0),
        x1.partialDerivative.map(t => t.toMat)
      )
    }
  }
  testGradientAndValue("add broadcasted - right")(Mat(Vec(1d)), 48d) {
    (m, doBackprop, cuda) =>
      Scope.root { implicit scope =>
        val x1 = param(lamp.saddle.fromMat(m, cuda))
        val x2 = param(lamp.saddle.fromMat(mat2x3 * 2, cuda))
        val L = (x2.+(x1)).sum
        if (doBackprop) {
          L.backprop()
        }
        (
          L.value.toMat.raw(0),
          x1.partialDerivative.map(t => t.toMat)
        )
      }
  }
  testGradientAndValue("add - right")(mat2x3, 63d) { (m, doBackprop, cuda) =>
    Scope.root { implicit scope =>
      val x1 = param(lamp.saddle.fromMat(m, cuda))
      val x2 = param(lamp.saddle.fromMat(mat2x3 * 2, cuda))
      val L = x2.+(x1).sum
      if (doBackprop) {
        L.backprop()
      }
      (
        L.value.toMat.raw(0),
        x1.partialDerivative.map(t => t.toMat)
      )
    }
  }
  testGradientAndValue("minus - left")(mat2x3, -21d) { (m, doBackprop, cuda) =>
    Scope.root { implicit scope =>
      val x1 = param(lamp.saddle.fromMat(m, cuda))
      val x2 = param(lamp.saddle.fromMat(mat2x3 * 2, cuda))
      val L = x1.-(x2).sum
      if (doBackprop) {
        L.backprop()
      }
      (
        L.value.toMat.raw(0),
        x1.partialDerivative.map(t => t.toMat)
      )
    }
  }
  testGradientAndValue("minus broadcasted - left")(mat1x1, -36d) {
    (m, doBackprop, cuda) =>
      Scope.root { implicit scope =>
        val x1 = param(lamp.saddle.fromMat(m, cuda))
        val x2 = param(lamp.saddle.fromMat(mat2x3 * 2, cuda))
        val L = x1.-(x2).sum
        if (doBackprop) {
          L.backprop()
        }
        (
          L.value.toMat.raw(0),
          x1.partialDerivative.map(t => t.toMat)
        )
      }
  }
  testGradientAndValue("minus broadcasted - right")(mat1x1, 36d) {
    (m, doBackprop, cuda) =>
      Scope.root { implicit scope =>
        val x1 = param(lamp.saddle.fromMat(m, cuda))
        val x2 = param(lamp.saddle.fromMat(mat2x3 * 2, cuda))
        val L = x2.-(x1).sum
        if (doBackprop) {
          L.backprop()
        }
        (
          L.value.toMat.raw(0),
          x1.partialDerivative.map(t => t.toMat)
        )
      }
  }
  testGradientAndValue("minus - right")(mat2x3, 21d) { (m, doBackprop, cuda) =>
    Scope.root { implicit scope =>
      val x1 = param(lamp.saddle.fromMat(m, cuda))
      val x2 = param(lamp.saddle.fromMat(mat2x3 * 2, cuda))
      val L = x2.-(x1).sum
      if (doBackprop) {
        L.backprop()
      }
      (
        L.value.toMat.raw(0),
        x1.partialDerivative.map(t => t.toMat)
      )
    }
  }
  testGradientAndValue("constmult")(mat2x3, 42d) { (m, doBackprop, cuda) =>
    Scope.root { implicit scope =>
      val x1 = param(lamp.saddle.fromMat(m, cuda))
      val L = x1.*(2d).sum
      if (doBackprop) {
        L.backprop()
      }
      (
        L.value.toMat.raw(0),
        x1.partialDerivative.map(t => t.toMat)
      )
    }
  }
  testGradientAndValue("cast to float")(mat2x3, 21d, 1e-2) {
    (m, doBackprop, cuda) =>
      Scope.root { implicit scope =>
        val x1 = param(lamp.saddle.fromMat(m, cuda))
        val L = x1.cast(lamp.SinglePrecision).sum
        if (doBackprop) {
          L.backprop()
        }
        (
          L.value.toMat.raw(0),
          x1.partialDerivative.map(t => t.toMat)
        )
      }
  }
  testGradientAndValue("constadd")(mat2x3, 33d) { (m, doBackprop, cuda) =>
    Scope.root { implicit scope =>
      val x1 = param(lamp.saddle.fromMat(m, cuda))
      val L = x1.+(2d).sum
      if (doBackprop) {
        L.backprop()
      }
      (
        L.value.toMat.raw(0),
        x1.partialDerivative.map(t => t.toMat)
      )
    }
  }
  testGradientAndValue("mult broadcasted - left")(mat1x1, 42d) {
    (m, doBackprop, cuda) =>
      Scope.root { implicit scope =>
        val x1 = param(lamp.saddle.fromMat(m, cuda))
        val x2 = param(lamp.saddle.fromMat(mat2x3 * 2, cuda))
        val L = x1.*(x2).sum
        if (doBackprop) {
          L.backprop()
        }
        (
          L.value.toMat.raw(0),
          x1.partialDerivative.map(t => t.toMat)
        )
      }
  }
  testGradientAndValue("mult broadcasted - right")(mat1x1, 42d) {
    (m, doBackprop, cuda) =>
      Scope.root { implicit scope =>
        val x1 = param(lamp.saddle.fromMat(m, cuda))
        val x2 = param(lamp.saddle.fromMat(mat2x3 * 2, cuda))
        val L = x2.*(x1).sum
        if (doBackprop) {
          L.backprop()
        }
        (
          L.value.toMat.raw(0),
          x1.partialDerivative.map(t => t.toMat)
        )
      }
  }
  testGradientAndValue("mult - right")(mat2x3, 182d) { (m, doBackprop, cuda) =>
    Scope.root { implicit scope =>
      val x1 = param(lamp.saddle.fromMat(m, cuda))
      val x2 = param(lamp.saddle.fromMat(mat2x3 * 2, cuda))
      val L = x2.*(x1).sum
      if (doBackprop) {
        L.backprop()
      }
      (
        L.value.toMat.raw(0),
        x1.partialDerivative.map(t => t.toMat)
      )
    }
  }

  testGradientAndValue("div - left")(mat2x3, 3d) { (m, doBackprop, cuda) =>
    Scope.root { implicit scope =>
      val x1 = param(lamp.saddle.fromMat(m, cuda))
      val x2 = param(lamp.saddle.fromMat(mat2x3 * 2, cuda))
      val L = x1./(x2).sum
      if (doBackprop) {
        L.backprop()
      }
      (
        L.value.toMat.raw(0),
        x1.partialDerivative.map(t => t.toMat)
      )
    }
  }
  testGradientAndValue("div broadcasted - left")(mat1x1, 1.225d) {
    (m, doBackprop, cuda) =>
      Scope.root { implicit scope =>
        val x1 = param(lamp.saddle.fromMat(m, cuda))
        val x2 = param(lamp.saddle.fromMat(mat2x3 * 2, cuda))
        val L = x1./(x2).sum
        if (doBackprop) {
          L.backprop()
        }
        (
          L.value.toMat.raw(0),
          x1.partialDerivative.map(t => t.toMat)
        )
      }
  }
  testGradientAndValue("div - right")(mat2x3, 12d) { (m, doBackprop, cuda) =>
    Scope.root { implicit scope =>
      val x1 = param(lamp.saddle.fromMat(m, cuda))
      val x2 = param(lamp.saddle.fromMat(mat2x3 * 2, cuda))
      val L = x2./(x1).sum
      if (doBackprop) {
        L.backprop()
      }
      (
        L.value.toMat.raw(0),
        x1.partialDerivative.map(t => t.toMat)
      )
    }
  }
  testGradientAndValue("min - left")(mat2x3, 21d) { (m, doBackprop, cuda) =>
    Scope.root { implicit scope =>
      val x1 = param(lamp.saddle.fromMat(m, cuda))
      val x2 = param(lamp.saddle.fromMat(mat2x3 * 2, cuda))
      val L = x1.minimum(x2).sum

      if (doBackprop) {
        L.backprop()
      }
      (
        L.value.toMat.raw(0),
        x1.partialDerivative.map(t => t.toMat)
      )
    }
  }
  testGradientAndValue("min - right")(mat2x3, 21d) { (m, doBackprop, cuda) =>
    Scope.root { implicit scope =>
      val x1 = param(lamp.saddle.fromMat(m, cuda))
      val x2 = param(lamp.saddle.fromMat(mat2x3 * 2, cuda))

      val L = x2.minimum(x1).sum
      if (doBackprop) {
        L.backprop()
      }
      (
        L.value.toMat.raw(0),
        x1.partialDerivative.map(t => t.toMat)
      )
    }
  }
  testGradientAndValue("max - left")(mat2x3, 42d) { (m, doBackprop, cuda) =>
    Scope.root { implicit scope =>
      val x1 = param(lamp.saddle.fromMat(m, cuda))
      val x2 = param(lamp.saddle.fromMat(mat2x3 * 2, cuda))
      val L = x1.maximum(x2).sum

      if (doBackprop) {
        L.backprop()
      }
      (
        L.value.toMat.raw(0),
        x1.partialDerivative.map(t => t.toMat)
      )
    }
  }
  testGradientAndValue("max - right")(mat2x3, 42d) { (m, doBackprop, cuda) =>
    Scope.root { implicit scope =>
      val x1 = param(lamp.saddle.fromMat(m, cuda))
      val x2 = param(lamp.saddle.fromMat(mat2x3 * 2, cuda))

      val L = x2.maximum(x1).sum
      if (doBackprop) {
        L.backprop()
      }
      (
        L.value.toMat.raw(0),
        x1.partialDerivative.map(t => t.toMat)
      )
    }
  }

  testGradientAndValue("mm - left")(mat2x3, 358d) { (m, doBackprop, cuda) =>
    Scope.root { implicit scope =>
      val x1 = param(lamp.saddle.fromMat(m, cuda))
      val x2 = param(lamp.saddle.fromMat(mat3x2 * 2, cuda))
      val L = x1.mm(x2).sum
      if (doBackprop) {
        L.backprop()
      }
      (
        L.value.toMat.raw(0),
        x1.partialDerivative.map(t => t.toMat)
      )
    }
  }
  testGradientAndValue("diag")(mat3x1, 6d) { (m, doBackprop, cuda) =>
    Scope.root { implicit scope =>
      val values = param(lamp.saddle.fromMat(m, cuda))

      val d = values.view(List(-1)).diag(0L)

      assert(d.value.toMat.numCols == 3)
      assert(d.value.toMat.numRows == 3)

      val L = d.sum
      if (doBackprop) {
        L.backprop()
      }
      (
        L.value.toMat.raw(0),
        values.partialDerivative.map(t => t.toMat)
      )
    }
  }
  testGradientAndValue("inv")(mat2x2, -0.5) { (m, doBackprop, cuda) =>
    Scope.root { implicit scope =>
      val values = param(lamp.saddle.fromMat(m, cuda))

      val i = values.inv
      assert(i.value.toMat.roundTo(4) == m.invert.roundTo(4))
      val L = i.sum
      if (doBackprop) {
        L.backprop()
      }
      (
        L.value.toMat.raw(0),
        values.partialDerivative.map(t => t.toMat)
      )
    }
  }
  testGradientAndValue("inv2")(mat3x3, 2d) { (m, doBackprop, cuda) =>
    Scope.root { implicit scope =>
      val values = param(lamp.saddle.fromMat(m, cuda))

      val i = values.inv
      assert(i.value.toMat.roundTo(4) == m.invert.roundTo(4))
      val L = i.sum
      if (doBackprop) {
        L.backprop()
      }
      (
        L.value.toMat.raw(0),
        values.partialDerivative.map(t => t.toMat)
      )
    }
  }
  testGradientAndValue("pinv")(mat3x3, 2d) { (m, doBackprop, cuda) =>
    Scope.root { implicit scope =>
      val values = param(lamp.saddle.fromMat(m, cuda))

      val i = values.pinv()
      val L = i.sum
      if (doBackprop) {
        L.backprop()
      }
      (
        L.value.toMat.raw(0),
        values.partialDerivative.map(t => t.toMat)
      )
    }
  }
  testGradientAndValue("pinv batch ")(mat3x3, 2d) { (m, doBackprop, cuda) =>
    Scope.root { implicit scope =>
      val t1 = lamp.saddle.fromMat(m, cuda)
      val values = param(STen.stack(List(t1, t1), 0))

      val i = values.pinv().select(0, 0)
      val L = i.sum
      if (doBackprop) {
        L.backprop()
      }
      (
        L.value.toMat.raw(0),
        values.partialDerivative.map(t => t.select(0, 0).toMat)
      )
    }
  }
  testGradientAndValueND("inv batch")(nd1x3x3, 0d) { (m, doBackprop, cuda) =>
    Scope.root { implicit scope =>
      val values = param(STen.owned(NDArray.tensorFromNDArray(m, cuda)))

      val i = values.inv
      val L = i.sum
      if (doBackprop) {
        L.backprop()
      }
      (
        L.value.toMat.raw(0),
        values.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
      )
    }
  }

  testGradientAndValue("sparse to dense")(mat3x1, 6d) { (m, doBackprop, cuda) =>
    Scope.root { implicit scope =>
      val values = param(lamp.saddle.fromMat(m, cuda))

      val idx =
        lamp.saddle.fromLongMat(Mat(Vec(1L, 0L, 1L), Vec(0L, 1L, 1L)).T, cuda)

      val dim = List(3L, 2L)

      val dense = Variable
        .sparseFromValueAndIndex(
          values = values.view(List(-1L)),
          indices = idx,
          dim = dim
        )
        .toDense

      assert(
        dense.value.toMat.roundTo(4) == Mat(
          Vec(0d, mat3x1.raw(0), 0d),
          Vec(mat3x1.raw(1), mat3x1.raw(2), 0d)
        ).roundTo(4)
      )

      val L = dense.sum
      if (doBackprop) {
        L.backprop()
      }
      (
        L.value.toMat.raw(0),
        values.partialDerivative.map(t => t.toMat)
      )
    }
  }
  testGradientAndValue("sparse mm - right")(mat2x3, 54d) {
    (m, doBackprop, cuda) =>
      Scope.root { implicit scope =>
        val x1 = param(lamp.saddle.fromMat(m, cuda))
        val x2 = {
          val idx =
            lamp.saddle.fromLongMat(Mat(Vec(1L, 0L), Vec(0L, 1L)), cuda)
          val values = lamp.saddle.fromVec(Vec(2d, 3d), cuda)
          val topt = if (cuda) STenOptions.d.cudaIndex(0) else STenOptions.d
          val sp = STen.sparse_coo(idx, values, List(3, 2), topt)
          const(sp)
        }
        val L = x2.mm(x1).sum
        if (doBackprop) {
          L.backprop()
        }
        (
          L.value.toMat.raw(0),
          x1.partialDerivative.map(t => t.toMat)
        )
      }
  }

  testGradientAndValue("mm - right")(mat2x3, 450d) { (m, doBackprop, cuda) =>
    Scope.root { implicit scope =>
      val x1 = param(lamp.saddle.fromMat(m, cuda))
      val x2 = param(lamp.saddle.fromMat(mat3x2 * 2, cuda))
      val L = x2.mm(x1).sum
      if (doBackprop) {
        L.backprop()
      }
      (
        L.value.toMat.raw(0),
        x1.partialDerivative.map(t => t.toMat)
      )
    }
  }
  testGradientAndValue("crossentropy - left")(mat2x3, -182.0) {
    (m, doBackprop, cuda) =>
      Scope.root { implicit scope =>
        val x1 = param(lamp.saddle.fromMat(m, cuda))
        val x2 = param(lamp.saddle.fromMat(mat2x3 * 2, cuda))

        val L = x1.crossEntropy(x2).sum
        if (doBackprop) {
          L.backprop()
        }
        (
          L.value.toMat.raw(0),
          x1.partialDerivative.map(t => t.toMat)
        )
      }
  }
  testGradientAndValue("crossentropy - right")(mat2x3, -182.0) {
    (m, doBackprop, cuda) =>
      Scope.root { implicit scope =>
        val x1 = param(lamp.saddle.fromMat(m, cuda))
        val x2 = param(lamp.saddle.fromMat(mat2x3 * 2, cuda))
        val L = x2.crossEntropy(x1).sum
        if (doBackprop) {
          L.backprop()
        }
        (
          L.value.toMat.raw(0),
          x1.partialDerivative.map(t => t.toMat)
        )
      }
  }

  testGradientAndValue("leakyrelu")(mat2x3_2, 13.5) { (m, doBackprop, cuda) =>
    Scope.root { implicit scope =>
      val x1 = param(lamp.saddle.fromMat(m, cuda))
      val L = x1.leakyRelu(0.5).sum
      if (doBackprop) {
        L.backprop()
      }
      (
        L.value.toMat.raw(0),
        x1.partialDerivative.map(t => t.toMat)
      )
    }
  }
  testGradientAndValue("relu")(mat2x3_2, 16d) { (m, doBackprop, cuda) =>
    Scope.root { implicit scope =>
      val x1 = param(lamp.saddle.fromMat(m, cuda))
      val L = x1.relu.sum
      if (doBackprop) {
        L.backprop()
      }
      (
        L.value.toMat.raw(0),
        x1.partialDerivative.map(t => t.toMat)
      )
    }
  }
  testGradientAndValue("gelu")(mat2x3_2, 15.7917) { (m, doBackprop, cuda) =>
    Scope.root { implicit scope =>
      val x1 = param(lamp.saddle.fromMat(m, cuda))
      val L = x1.gelu.sum
      if (doBackprop) {
        L.backprop()
      }
      (
        L.value.toMat.raw(0),
        x1.partialDerivative.map(t => t.toMat)
      )
    }
  }
  testGradientAndValue("sigmoid")(mat2x3_2, 4.1111) { (m, doBackprop, cuda) =>
    Scope.root { implicit scope =>
      val x1 = param(lamp.saddle.fromMat(m, cuda))
      val L = x1.sigmoid.sum
      if (doBackprop) {
        L.backprop()
      }
      (
        L.value.toMat.raw(0),
        x1.partialDerivative.map(t => t.toMat)
      )
    }
  }
  testGradientAndValue("hardswish")(mat2x3_2, 15.7700) {
    (m, doBackprop, cuda) =>
      Scope.root { implicit scope =>
        val x1 = param(lamp.saddle.fromMat(m, cuda) + 0.1)
        val L = x1.hardSwish.sum
        if (doBackprop) {
          L.backprop()
        }
        (
          L.value.toMat.raw(0),
          x1.partialDerivative.map(t => t.toMat)
        )
      }
  }
  testGradientAndValue("exp")(mat2x3_2, 579.7027406974902) {
    (m, doBackprop, cuda) =>
      Scope.root { implicit scope =>
        val x1 = param(lamp.saddle.fromMat(m, cuda))
        val L = x1.exp.sum
        if (doBackprop) {
          L.backprop()
        }
        (
          L.value.toMat.raw(0),
          x1.partialDerivative.map(t => t.toMat)
        )
      }
  }
  testGradientAndValue("logdet")(mat2x2PD, 1.3863) { (m, doBackprop, cuda) =>
    Scope.root { implicit scope =>
      val x1 = param(lamp.saddle.fromMat(m, cuda))
      val L = x1.logdet.sum
      if (doBackprop) {
        L.backprop()
      }
      (
        L.value.toMat.raw(0),
        x1.partialDerivative.map(t => t.toMat)
      )
    }
  }
  // won't pass because the grad is symmetrized
  // testGradientAndValue("cholesky")(mat2x2PD, 9.7073) { (m, doBackprop, cuda) =>
  //   Scope.root { implicit scope =>
  //     val x1 = param(lamp.saddle.fromMat(m, cuda))
  //     val L = x1.cholesky().sum
  //     if (doBackprop) {
  //       L.backprop()
  //     }
  //     (
  //       L.value.toMat.raw(0),
  //       x1.partialDerivative.map(t => t.toMat)
  //     )
  //   }
  // }
  testGradientAndValue("cholesky solve, b")(mat2x2, 58.2500) {
    (m, doBackprop, cuda) =>
      Scope.root { implicit scope =>
        val l = param(lamp.saddle.fromMat(mat2x2PD).choleskyLower)
        val x1 = param(lamp.saddle.fromMat(m, cuda))
        val L = x1.choleskySolve(l, false).sum
        if (doBackprop) {
          L.backprop()
        }
        (
          L.value.toMat.raw(0),
          x1.partialDerivative.map(t => t.toMat)
        )
      }
  }
  // won't pass because the grad is symmetrized
  // testGradientAndValue("cholesky solve, f")(mat2x2PDCholeskyFactor, 6.25d) {
  //   (m, doBackprop, cuda) =>
  //     Scope.root { implicit scope =>
  //       val x1 = param(lamp.saddle.fromMat(m, cuda))
  //       val b = param(STen.eye(2))
  //       val L = b.choleskySolve(x1, false).sum
  //       if (doBackprop) {
  //         L.backprop()
  //       }
  //       (
  //         L.value.toMat.raw(0),
  //         x1.partialDerivative.map(t => t.toMat)
  //       )
  //     }
  // }
  testGradientAndValue("log")(mat2x3, 6.579251212010101) {
    (m, doBackprop, cuda) =>
      Scope.root { implicit scope =>
        val x1 = param(lamp.saddle.fromMat(m, cuda))
        val L = x1.log.sum
        if (doBackprop) {
          L.backprop()
        }
        (
          L.value.toMat.raw(0),
          x1.partialDerivative.map(t => t.toMat)
        )
      }
  }
  testGradientAndValue("log1p")(mat2x3, 8.5252) { (m, doBackprop, cuda) =>
    Scope.root { implicit scope =>
      val x1 = param(lamp.saddle.fromMat(m, cuda))
      val L = x1.log1p.sum
      if (doBackprop) {
        L.backprop()
      }
      (
        L.value.toMat.raw(0),
        x1.partialDerivative.map(t => t.toMat)
      )
    }
  }
  testGradientAndValue("softplus")(mat2x3, 21.0000) { (m, doBackprop, cuda) =>
    Scope.root { implicit scope =>
      val x1 = param(lamp.saddle.fromMat(m, cuda))
      val L = x1.softplus(beta = 2d, threshold = 0d).sum
      if (doBackprop) {
        L.backprop()
      }
      (
        L.value.toMat.raw(0),
        x1.partialDerivative.map(t => t.toMat)
      )
    }
  }
  testGradientAndValue("cross left")(mat2x3, 28.0000) { (m, doBackprop, cuda) =>
    Scope.root { implicit scope =>
      val x1 = param(lamp.saddle.fromMat(m, cuda))
      val x2 = const(lamp.saddle.fromMat(mat2x3_2, cuda))
      val L = x1.cross(x2, 1).sum
      if (doBackprop) {
        L.backprop()
      }
      (
        L.value.toMat.raw(0),
        x1.partialDerivative.map(t => t.toMat)
      )
    }
  }
  testGradientAndValue("cross right")(mat2x3, -28.0000) {
    (m, doBackprop, cuda) =>
      Scope.root { implicit scope =>
        val x1 = param(lamp.saddle.fromMat(m, cuda))
        val x2 = const(lamp.saddle.fromMat(mat2x3_2, cuda))
        val L = x2.cross(x1, 1).sum
        if (doBackprop) {
          L.backprop()
        }
        (
          L.value.toMat.raw(0),
          x1.partialDerivative.map(t => t.toMat)
        )
      }
  }
  testGradientAndValue("sin")(mat2x3_2, -0.27259082747648367) {
    (m, doBackprop, cuda) =>
      Scope.root { implicit scope =>
        val x1 = param(lamp.saddle.fromMat(m, cuda))
        val L = x1.sin.sum
        if (doBackprop) {
          L.backprop()
        }
        (
          L.value.toMat.raw(0),
          x1.partialDerivative.map(t => t.toMat)
        )
      }
  }
  testGradientAndValue("cos")(mat2x3_2, -0.2756481760294678) {
    (m, doBackprop, cuda) =>
      Scope.root { implicit scope =>
        val x1 = param(lamp.saddle.fromMat(m, cuda))
        val L = x1.cos.sum
        if (doBackprop) {
          L.backprop()
        }
        (
          L.value.toMat.raw(0),
          x1.partialDerivative.map(t => t.toMat)
        )
      }
  }
  testGradientAndValue("tan")(mat2x3_2, -8.71433661097161) {
    (m, doBackprop, cuda) =>
      Scope.root { implicit scope =>
        val x1 = param(lamp.saddle.fromMat(m, cuda))
        val L = x1.tan.sum
        if (doBackprop) {
          L.backprop()
        }
        (
          L.value.toMat.raw(0),
          x1.partialDerivative.map(t => t.toMat)
        )
      }
  }

  testGradientAndValue("atan")(mat2x3_2, 3.02402707945215) {
    (m, doBackprop, cuda) =>
      Scope.root { implicit scope =>
        val x1 = param(lamp.saddle.fromMat(m, cuda))
        val L = x1.atan.sum
        if (doBackprop) {
          L.backprop()
        }
        (
          L.value.toMat.raw(0),
          x1.partialDerivative.map(t => t.toMat)
        )
      }
  }
  testGradientAndValue("capped exp")(mat3x1, 2.6065) { (m, doBackprop, cuda) =>
    Scope.root { implicit scope =>
      val x1 = param(lamp.saddle.fromMat(m, cuda))
      val out = new CappedShiftedNegativeExponential(scope, x1, 2.5d).value
      assert(
        out.value.toMat.roundTo(4) == Mat(Vec(1d, 1d, math.exp(-0.5)))
          .roundTo(4)
      )
      val L = out.sum
      if (doBackprop) {
        L.backprop()
      }
      (
        L.value.toMat.raw(0),
        x1.partialDerivative.map(t => t.toMat)
      )
    }
  }
  testGradientAndValue("pow")(mat2x3_2, 91d) { (m, doBackprop, cuda) =>
    Scope.root { implicit scope =>
      val x1 = param(lamp.saddle.fromMat(m, cuda))
      val L = x1.pow(2d).sum
      if (doBackprop) {
        L.backprop()
      }
      (
        L.value.toMat.raw(0),
        x1.partialDerivative.map(t => t.toMat)
      )
    }
  }
  testGradientAndValue("euclidean distance wrt a")(mat2x3_2, 10d) {
    (m, doBackprop, cuda) =>
      Scope.root { implicit scope =>
        val x1 = param(lamp.saddle.fromMat(m, cuda))
        val out =
          x1.euclideanDistance(const(lamp.saddle.fromMat(mat2x3, cuda)), 1)
        assert(out.shape == List(2, 1))
        assert(out.value.toMat.roundTo(4) == Mat(Vec(2d, 8d)))
        val L = out.sum
        if (doBackprop) {
          L.backprop()
        }
        (
          L.value.toMat.raw(0),
          x1.partialDerivative.map(t => t.toMat)
        )
      }
  }
  testGradientAndValue("euclidean distance wrt b")(mat2x3_2, 10d) {
    (m, doBackprop, cuda) =>
      Scope.root { implicit scope =>
        val x1 = param(lamp.saddle.fromMat(m, cuda))
        val out =
          const(lamp.saddle.fromMat(mat2x3, cuda)).euclideanDistance(x1, 1)
        assert(out.shape == List(2, 1))
        assert(out.value.toMat.roundTo(4) == Mat(Vec(2d, 8d)))
        val L = out.sum
        if (doBackprop) {
          L.backprop()
        }
        (
          L.value.toMat.raw(0),
          x1.partialDerivative.map(t => t.toMat)
        )
      }
  }
  testGradientAndValue("pow  2")(mat1x1, 11d) { (m, doBackprop, cuda) =>
    Scope.root { implicit scope =>
      val x1 = param(lamp.saddle.fromMat(m, cuda))
      val x2 = param(lamp.saddle.fromMat(mat2x3_2, cuda))
      val L = x2.pow(x1).sum
      if (doBackprop) {
        L.backprop()
      }
      (
        L.value.toMat.raw(0),
        x1.partialDerivative.map(t => t.toMat)
      )
    }
  }
  testGradientAndValue("tanh")(mat2x3_2, 2.1981) { (m, doBackprop, cuda) =>
    Scope.root { implicit scope =>
      val x1 = param(lamp.saddle.fromMat(m, cuda))
      val L = x1.tanh.sum
      if (doBackprop) {
        L.backprop()
      }
      (
        L.value.toMat.raw(0),
        x1.partialDerivative.map(t => t.toMat)
      )
    }
  }
  testGradientAndValue("where true branch")(mat2x3_2, 21d) {
    (m, doBackprop, cuda) =>
      Scope.root { implicit scope =>
        val x1 = param(lamp.saddle.fromMat(m, cuda))
        val x2 = param(lamp.saddle.fromMat(mat2x3, cuda))
        val L =
          Variable.where(lamp.saddle.fromMat(mat2x3_2).equ(2.0), x1, x2).sum
        if (doBackprop) {
          L.backprop()
        }
        (
          L.value.toMat.raw(0),
          x1.partialDerivative.map(t => t.toMat)
        )
      }
  }
  testGradientAndValue("where false branch")(mat2x3, 21d) {
    (m, doBackprop, cuda) =>
      Scope.root { implicit scope =>
        val x1 = param(lamp.saddle.fromMat(mat2x3_2, cuda))
        val x2 = param(lamp.saddle.fromMat(m, cuda))
        val L =
          Variable.where(lamp.saddle.fromMat(mat2x3_2).equ(2.0), x1, x2).sum
        if (doBackprop) {
          L.backprop()
        }
        (
          L.value.toMat.raw(0),
          x2.partialDerivative.map(t => t.toMat)
        )
      }
  }
  testGradientAndValue("softmax")(mat2x3_2, -22.441910257332836) {
    (m, doBackprop, cuda) =>
      Scope.root { implicit scope =>
        val x1 = param(lamp.saddle.fromMat(m, cuda))
        val L = x1.logSoftMax(dim = 1).sum
        if (doBackprop) {
          L.backprop()
        }
        (
          L.value.toMat.raw(0),
          x1.partialDerivative.map(t => t.toMat)
        )
      }
  }
  testGradientAndValue("squaredFrobenius")(mat2x3_2, 91d) {
    (m, doBackprop, cuda) =>
      Scope.root { implicit scope =>
        val x1 = param(lamp.saddle.fromMat(m, cuda))
        val L = x1.squaredFrobenius.sum
        if (doBackprop) {
          L.backprop()
        }
        (
          L.value.toMat.raw(0),
          x1.partialDerivative.map(t => t.toMat)
        )
      }
  }
  testGradientAndValue("transpose")(mat2x3_2, 11d) { (m, doBackprop, cuda) =>
    Scope.root { implicit scope =>
      val x1 = param(lamp.saddle.fromMat(m, cuda))
      val L = x1.t.sum
      if (doBackprop) {
        L.backprop()
      }
      (
        L.value.toMat.raw(0),
        x1.partialDerivative.map(t => t.toMat)
      )
    }
  }
  testGradientAndValue("mean")(mat2x3_2, 1.8333d) { (m, doBackprop, cuda) =>
    Scope.root { implicit scope =>
      val x1 = param(lamp.saddle.fromMat(m, cuda))
      val L = x1.mean(List(0, 1))
      if (doBackprop) {
        L.backprop()
      }
      (
        L.value.toMat.raw(0),
        x1.partialDerivative.map(t => t.toMat)
      )
    }
  }
  testGradientAndValue("norm2")(mat2x3_2, 9.5394) { (m, doBackprop, cuda) =>
    Scope.root { implicit scope =>
      val x1 = param(lamp.saddle.fromMat(m, cuda))
      val L = x1.norm2(List(0, 1), true)
      if (doBackprop) {
        L.backprop()
      }
      (
        L.value.toMat.raw(0),
        x1.partialDerivative.map(t => t.toMat)
      )
    }
  }
  testGradientAndValue("mse loss")(mat3x1, 1d) { (m, doBackprop, cuda) =>
    Scope.root { implicit scope =>
      val x1 = param(lamp.saddle.fromMat(m, cuda))
      val x2 = lamp.saddle.fromMat(mat3x1_2, cuda)
      val L = x1.mseLoss(x2.squeeze).sum
      if (doBackprop) {
        L.backprop()
      }
      (
        L.value.toMat.raw(0),
        x1.partialDerivative.map(t => t.toMat)
      )
    }
  }
  testGradientAndValue("l1 loss")(mat3x1, 0.5) { (m, doBackprop, cuda) =>
    Scope.root { implicit scope =>
      val x1 = param(lamp.saddle.fromMat(m, cuda))
      val x2 = lamp.saddle.fromMat(mat3x1_2, cuda)
      val L = x1.smoothL1Loss(x2.squeeze).sum
      if (doBackprop) {
        L.backprop()
      }
      (
        L.value.toMat.raw(0),
        x1.partialDerivative.map(t => t.toMat)
      )
    }
  }
  testGradientAndValue("l2 logistic regression loss")(
    mat2x3_2,
    151.0000008318073
  ) { (m, doBackprop, _) =>
    Scope.root { implicit scope =>
      val w = param(lamp.saddle.fromMat(m))
      val data = const(lamp.saddle.fromMat(mat3x2))
      val y = const(lamp.saddle.fromMat(mat.ident(3)))
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
        L.value.toMat.raw(0),
        w.partialDerivative.map(t => t.toMat)
      )
    }
  }
  testGradientAndValue("l2 logistic regression loss - bce loss")(
    mat2x2,
    54.5067
  ) { (m, doBackprop, _) =>
    Scope.root { implicit scope =>
      val w = param(lamp.saddle.fromMat(m))
      val data = const(lamp.saddle.fromMat(mat3x2))
      val y =
        const(lamp.saddle.fromMat(Mat(Vec(0d, 1d, 0.5d), Vec(0d, 0.5, 1d))))
      val classWeights = STen.ones(List(1, 2), w.value.options)
      val L =
        ((data
          .mm(w))
          .binaryCrossEntropyWithLogitsLoss(y.value, Some(classWeights), Sum))
      if (doBackprop) {
        L.backprop()
      }
      (
        L.value.toMat.raw(0),
        w.partialDerivative.map(t => t.toMat)
      )
    }
  }
  testGradientAndValue("l2 logistic regression loss - bce loss mean")(
    mat2x2,
    9.0845
  ) { (m, doBackprop, _) =>
    Scope.root { implicit scope =>
      val w = param(lamp.saddle.fromMat(m))
      val data = const(lamp.saddle.fromMat(mat3x2))
      val y =
        const(lamp.saddle.fromMat(Mat(Vec(0d, 1d, 0.5d), Vec(0d, 0.5, 1d))))
      val classWeights = STen.ones(List(1, 2), w.value.options)
      val L =
        ((data
          .mm(w))
          .binaryCrossEntropyWithLogitsLoss(y.value, Some(classWeights), Mean))
      if (doBackprop) {
        L.backprop()
      }
      (
        L.value.toMat.raw(0),
        w.partialDerivative.map(t => t.toMat)
      )
    }
  }
  testGradientAndValue("l2 logistic regression loss - nll_loss")(
    mat2x3_2,
    151.0000008318073
  ) { (m, doBackprop, _) =>
    Scope.root { implicit scope =>
      val w = param(lamp.saddle.fromMat(m))
      val data = const(lamp.saddle.fromMat(mat3x2))
      val y =
        const(lamp.saddle.fromLongMat(Mat(Vec(0L, 1L, 2L))).squeeze)
      val classWeights = STen.ones(List(3), w.value.options)
      val L =
        ((data
          .mm(w))
          .logSoftMax(dim = 1)
          .nllLoss(y.value, classWeights, Sum) + w.squaredFrobenius)
      if (doBackprop) {
        L.backprop()
      }
      (
        L.value.toMat.raw(0),
        w.partialDerivative.map(t => t.toMat)
      )
    }
  }
  testGradientAndValue("l2 logistic regression loss - nll_loss unreduced")(
    mat2x3_2,
    151.0000008318073
  ) { (m, doBackprop, _) =>
    Scope.root { implicit scope =>
      val w = param(lamp.saddle.fromMat(m))
      val data = const(lamp.saddle.fromMat(mat3x2))
      val y =
        const(lamp.saddle.fromLongMat(Mat(Vec(0L, 1L, 2L))).squeeze)
      val classWeights = STen.ones(List(3), w.value.options)
      val L =
        ((data
          .mm(w))
          .logSoftMax(dim = 1)
          .nllLoss(y.value, classWeights, NoReduction)
          .sum + w.squaredFrobenius)
      if (doBackprop) {
        L.backprop()
      }
      (
        L.value.toMat.raw(0),
        w.partialDerivative.map(t => t.toMat)
      )
    }
  }
  testGradientAndValue("weight norm - wrt g")(mat2x3.row(Array(0)), 12.7279) {
    (m, doBackprop, cuda) =>
      Scope.root { implicit scope =>
        val v = param(lamp.saddle.fromMat(mat.ones(2, 3), cuda))
        val g = param(lamp.saddle.fromMat(m, cuda))
        val L = new WeightNorm(scope, v, g, 0).value.sum
        if (doBackprop) {
          L.backprop()
        }
        (
          L.value.toMat.raw(0),
          g.partialDerivative.map(t => t.toMat)
        )
      }
  }
  testGradientAndValue("weight norm - wrt v")(mat2x3, 4.1500) {
    (m, doBackprop, cuda) =>
      Scope.root { implicit scope =>
        val v = param(lamp.saddle.fromMat(m, cuda))
        val g = param(lamp.saddle.fromMat(mat.ones(1, 3), cuda))
        val L = new WeightNorm(scope, v, g, 0).value.sum
        if (doBackprop) {
          L.backprop()
        }
        (
          L.value.toMat.raw(0),
          v.partialDerivative.map(t => t.toMat)
        )
      }
  }
  testGradientAndValueND("mask-fill")(nd1x2x2, 5d) { (m, doBackprop, cuda) =>
    Scope.root { implicit scope =>
      val input =
        param(STen.owned(NDArray.tensorFromNDArray(m, cuda)))
      val mask = {
        val q = STen.owned(
          NDArray.tensorFromNDArray(
            NDArray(Array(1d, 0d, 0d, 0d), List(1, 2, 2)),
            cuda
          )
        )
        val sc = STen.scalarDouble(1d, q.options)
        param(q.equ(sc))
      }

      val output = new MaskFill(scope, input, mask, 2d).value

      val L = output.sum
      if (doBackprop) {
        L.backprop()
      }
      (
        L.value.toMat.raw(0),
        input.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
      )
    }
  }
  testGradientAndValueND("mask-select")(nd1x2x2, 2d) { (m, doBackprop, cuda) =>
    Scope.root { implicit scope =>
      val input =
        param(STen.owned(NDArray.tensorFromNDArray(m, cuda)))
      val mask = {
        val q = STen.owned(
          NDArray.tensorFromNDArray(
            NDArray(Array(1d, 0d, 0d, 1d), List(1, 2, 2)),
            cuda
          )
        )
        val sc = STen.scalarDouble(1d, q.options)
        param(q.equ(sc))
      }

      val output = new MaskSelect(scope, input.flatten, mask.flatten).value
      assert(output.shape == List(2))

      val L = output.sum
      if (doBackprop) {
        L.backprop()
      }
      (
        L.value.toMat.raw(0),
        input.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
      )
    }
  }
  testGradientAndValueND("index_fill")(nd1x2x2, 6d) { (m, doBackprop, cuda) =>
    Scope.root { implicit scope =>
      val input =
        param(STen.owned(NDArray.tensorFromNDArray(m, cuda)))
      val index =
        param(
          STen.owned(
            NDArray.tensorFromLongNDArray(NDArray(Array(1L), List(1)), cuda)
          )
        )

      val output = new IndexFill(scope, input, 1L, index, 2d).value

      val L = output.sum
      if (doBackprop) {
        L.backprop()
      }
      (
        L.value.toMat.raw(0),
        input.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
      )
    }
  }
  testGradientAndValueND("expand as")(nd1x2x2, 16d) { (m, doBackprop, cuda) =>
    Scope.root { implicit scope =>
      val input =
        param(STen.owned(NDArray.tensorFromNDArray(m, cuda)))
      val other =
        STen.owned(
          NDArray.tensorFromNDArray(
            NDArray(
              org.saddle.vec.zeros(16).toArray,
              List(2, 2, 2, 2)
            ),
            cuda
          )
        )
      val output = input.expandAs(other)
      assert(output.shape == other.shape)
      val L = output.sum
      if (doBackprop) {
        L.backprop()
      }
      (
        L.value.toMat.raw(0),
        input.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
      )
    }
  }
  testGradientAndValue("scatter sum")(mat2x3, 21d) { (m, doBackprop, cuda) =>
    Scope.root { implicit scope =>
      val input =
        param(lamp.saddle.fromMat(m, cuda))
      val index =
        param(
          STen.owned(
            NDArray.tensorFromLongNDArray(
              NDArray(
                Array(0L, 0L, 1L, 0L, 1L, 1L),
                List(2, 3)
              ),
              cuda
            )
          )
        )
      val output = input.scatterAdd(index, 0, 2)
      val L = output.sum
      if (doBackprop) {
        L.backprop()
      }
      (
        L.value.toMat.raw(0),
        input.partialDerivative.map(t => t.toMat)
      )
    }
  }

  testGradientAndValue("variance")(mat2x3, 8d) { (m, doBackprop, cuda) =>
    Scope.root { implicit scope =>
      val input =
        param(lamp.saddle.fromMat(m, cuda))

      val output = input.variance(List(1))
      val L = output.sum
      if (doBackprop) {
        L.backprop()
      }
      (
        L.value.toMat.raw(0),
        input.partialDerivative.map(t => t.toMat)
      )
    }
  }
  testGradientAndValue("index sum")(mat2x3, 21d) { (m, doBackprop, cuda) =>
    Scope.root { implicit scope =>
      val input =
        param(lamp.saddle.fromMat(m, cuda))
      val index =
        param(
          STen.owned(
            NDArray.tensorFromLongNDArray(
              NDArray(
                Array(1L, 1L),
                List(2)
              ),
              cuda
            )
          )
        )
      val output = input.indexAdd(index, 0, 2)
      assert(
        output.value.toMat.roundTo(4) == Mat(
          Vec(0d, 0d, 0d),
          Vec(3d, 7d, 11d)
        ).T
      )
      val L = output.sum
      if (doBackprop) {
        L.backprop()
      }
      (
        L.value.toMat.raw(0),
        input.partialDerivative.map(t => t.toMat)
      )
    }
  }
  testGradientAndValue("index add by target")(mat2x3, 27d) {
    (m, doBackprop, cuda) =>
      Scope.root { implicit scope =>
        val input =
          param(lamp.saddle.fromMat(m, cuda))
        val src = param(lamp.saddle.fromMat(mat3x1.T, cuda))
        val index =
          param(
            STen.owned(
              NDArray.tensorFromLongNDArray(
                NDArray(
                  Array(0L),
                  List(1)
                ),
                cuda
              )
            )
          )
        val output = input.indexAddFromSource(index, 0, src)
        assert(
          output.value.toMat.roundTo(4) == Mat(
            Vec(2d, 5d, 8d),
            Vec(2d, 4d, 6d)
          ).T
        )
        val L = output.sum
        if (doBackprop) {
          L.backprop()
        }
        (
          L.value.toMat.raw(0),
          input.partialDerivative.map(t => t.toMat)
        )
      }
  }
  testGradientAndValue("index add by src")(mat1x3, 27d) {
    (m, doBackprop, cuda) =>
      Scope.root { implicit scope =>
        val input =
          param(lamp.saddle.fromMat(mat2x3, cuda))
        val src = param(lamp.saddle.fromMat(m, cuda))
        val index =
          param(
            STen.owned(
              NDArray.tensorFromLongNDArray(
                NDArray(
                  Array(0L),
                  List(1)
                ),
                cuda
              )
            )
          )
        val output = input.indexAddFromSource(index, 0, src)
        assert(
          output.value.toMat.roundTo(4) == Mat(
            Vec(2d, 5d, 8d),
            Vec(2d, 4d, 6d)
          ).T
        )
        val L = output.sum
        if (doBackprop) {
          L.backprop()
        }
        (
          L.value.toMat.raw(0),
          src.partialDerivative.map(t => t.toMat)
        )
      }
  }
  testGradientAndValue("repeat interleave")(mat2x3, 54d) {
    (m, doBackprop, cuda) =>
      Scope.root { implicit scope =>
        val input =
          param(lamp.saddle.fromMat(m, cuda))
        val index =
          param(
            STen.owned(
              NDArray.tensorFromLongNDArray(
                NDArray(
                  Array(2L, 3L),
                  List(2)
                ),
                cuda
              )
            )
          )
        val output = input.repeatInterleave(index, 0)
        assert(
          output.value.toMat.roundTo(4) == Mat(
            Vec(1d, 1d, 2d, 2d, 2d),
            Vec(3d, 3d, 4d, 4d, 4d),
            Vec.apply[Double](5d, 5d, 6d, 6d, 6d)
          )
        )
        val L = output.sum
        if (doBackprop) {
          L.backprop()
        }
        (
          L.value.toMat.raw(0),
          input.partialDerivative.map(t => t.toMat)
        )
      }
  }
  testGradientAndValueND("index_select")(nd1x2x2, 6d) { (m, doBackprop, cuda) =>
    Scope.root { implicit scope =>
      val input =
        param(STen.owned(NDArray.tensorFromNDArray(m, cuda)))
      val index =
        param(
          STen.owned(
            NDArray
              .tensorFromLongNDArray(NDArray(Array(1L, 1L, 1L), List(3)), cuda)
          )
        )

      val output = new IndexSelect(scope, input, 1L, index).value

      val L = output.sum
      if (doBackprop) {
        L.backprop()
      }
      (
        L.value.toMat.raw(0),
        input.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
      )
    }
  }
  testGradientAndValueND("conv1d - wrt weights")(nd1x2x2, 30d) {
    (m, doBackprop, cuda) =>
      Scope.root { implicit scope =>
        val input =
          param(STen.owned(NDArray.tensorFromNDArray(nd1x2x3, cuda)))
        val weight = param(STen.owned(NDArray.tensorFromNDArray(m, cuda)))

        val bias = param(lamp.saddle.fromVec(vec.ones(1), cuda))
        val output =
          new Convolution(
            scope = scope,
            input = input,
            weight = weight,
            bias = bias,
            stride = Array(1),
            padding = Array(0),
            dilation = Array(1),
            transposed = false,
            outputPadding = Array(0),
            groups = 1L
          ).value

        val L = output.sum
        if (doBackprop) {
          L.backprop()
        }
        (
          L.value.toMat.raw(0),
          weight.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
        )
      }
  }
  testGradientAndValueND("conv1d - wrt input")(nd1x2x3, 30d) {
    (m, doBackprop, cuda) =>
      Scope.root { implicit scope =>
        val input =
          param(STen.owned(NDArray.tensorFromNDArray(m, cuda)))
        val weight = param(STen.owned(NDArray.tensorFromNDArray(nd1x2x2, cuda)))

        val bias = param(lamp.saddle.fromVec(vec.ones(1), cuda))

        val output =
          new Convolution(
            scope = scope,
            input = input,
            weight = weight,
            bias = bias,
            stride = Array(1),
            padding = Array(0),
            dilation = Array(1),
            transposed = false,
            outputPadding = Array(0),
            groups = 1L
          ).value

        val L = output.sum
        if (doBackprop) {
          L.backprop()
        }
        (
          L.value.toMat.raw(0),
          input.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
        )
      }
  }
  testGradientAndValueND("conv1d - padded - wrt weights")(nd1x2x2, 46d) {
    (m, doBackprop, cuda) =>
      Scope.root { implicit scope =>
        val input =
          param(STen.owned(NDArray.tensorFromNDArray(nd1x2x3, cuda)))
        val weight =
          param(STen.owned(NDArray.tensorFromNDArray(m, cuda)))

        val bias = param(lamp.saddle.fromVec(vec.ones(1), cuda))
        val output =
          new Convolution(
            scope = scope,
            input = input,
            weight = weight,
            bias = bias,
            stride = Array(1),
            padding = Array(1),
            dilation = Array(1),
            transposed = false,
            outputPadding = Array(0),
            groups = 1L
          ).value

        val L = output.sum
        if (doBackprop) {
          L.backprop()
        }
        (
          L.value.toMat.raw(0),
          weight.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
        )
      }
  }
  testGradientAndValueND("conv1d -padded - wrt input")(nd1x2x3, 46d) {
    (m, doBackprop, cuda) =>
      Scope.root { implicit scope =>
        val input =
          param(STen.owned(NDArray.tensorFromNDArray(m, cuda)))
        val weight = param(STen.owned(NDArray.tensorFromNDArray(nd1x2x2, cuda)))

        val bias = param(lamp.saddle.fromVec(vec.ones(1), cuda))
        val output =
          new Convolution(
            scope = scope,
            input = input,
            weight = weight,
            bias = bias,
            stride = Array(1),
            padding = Array(1),
            dilation = Array(1),
            transposed = false,
            outputPadding = Array(0),
            groups = 1L
          ).value

        val L = output.sum
        if (doBackprop) {
          L.backprop()
        }
        (
          L.value.toMat.raw(0),
          input.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
        )
      }
  }
  testGradientAndValueND("conv1d - stride-2 - wrt weights")(nd1x2x2, 23d) {
    (m, doBackprop, cuda) =>
      Scope.root { implicit scope =>
        val input =
          param(STen.owned(NDArray.tensorFromNDArray(nd1x2x3, cuda)))
        val weight = param(STen.owned(NDArray.tensorFromNDArray(m, cuda)))

        val bias = param(lamp.saddle.fromVec(vec.ones(1), cuda))
        val output =
          new Convolution(
            scope = scope,
            input = input,
            weight = weight,
            bias = bias,
            stride = Array(2),
            padding = Array(1),
            dilation = Array(1),
            transposed = false,
            outputPadding = Array(0),
            groups = 1L
          ).value

        val L = output.sum
        if (doBackprop) {
          L.backprop()
        }
        (
          L.value.toMat.raw(0),
          weight.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
        )
      }
  }
  testGradientAndValueND("conv1d -stride-2 - wrt input")(nd1x2x3, 23d) {
    (m, doBackprop, cuda) =>
      Scope.root { implicit scope =>
        val input =
          param(STen.owned(NDArray.tensorFromNDArray(m, cuda)))
        val weight = param(STen.owned(NDArray.tensorFromNDArray(nd1x2x2, cuda)))

        val bias = param(lamp.saddle.fromVec(vec.ones(1), cuda))
        val output =
          new Convolution(
            scope = scope,
            input = input,
            weight = weight,
            bias = bias,
            stride = Array(2),
            padding = Array(1),
            dilation = Array(1),
            transposed = false,
            outputPadding = Array(0),
            groups = 1L
          ).value

        val L = output.sum
        if (doBackprop) {
          L.backprop()
        }
        (
          L.value.toMat.raw(0),
          input.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
        )
      }
  }
  testGradientAndValueND("conv1d -stride-2 - wrt bias")(ndx1, 23d) {
    (m, doBackprop, cuda) =>
      Scope.root { implicit scope =>
        val input =
          param(STen.owned(NDArray.tensorFromNDArray(nd1x2x3, cuda)))
        val weight = param(STen.owned(NDArray.tensorFromNDArray(nd1x2x2, cuda)))

        val bias = param(STen.owned(NDArray.tensorFromNDArray(m, cuda)))
        val output =
          new Convolution(
            scope = scope,
            input = input,
            weight = weight,
            bias = bias,
            stride = Array(2),
            padding = Array(1),
            dilation = Array(1),
            transposed = false,
            outputPadding = Array(0),
            groups = 1L
          ).value

        val L = output.sum
        if (doBackprop) {
          L.backprop()
        }
        (
          L.value.toMat.raw(0),
          bias.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
        )
      }
  }
  testGradientAndValueND("conv2d - wrt weights")(nd1x2x2x2, 276d) {
    (m, doBackprop, cuda) =>
      Scope.root { implicit scope =>
        val input =
          param(STen.owned(NDArray.tensorFromNDArray(nd1x2x3x3, cuda)))
        val weight = param(STen.owned(NDArray.tensorFromNDArray(m, cuda)))

        val bias = param(lamp.saddle.fromVec(vec.ones(1), cuda))
        val output =
          new Convolution(
            scope = scope,
            input = input,
            weight = weight,
            bias = bias,
            stride = Array(1, 1),
            padding = Array(0, 0),
            dilation = Array(1, 1),
            transposed = false,
            outputPadding = Array(0, 0),
            groups = 1L
          ).value

        val L = output.sum
        if (doBackprop) {
          L.backprop()
        }
        (
          L.value.toMat.raw(0),
          weight.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
        )
      }
  }
  testGradientAndValueND("conv2d - wrt weights - groups", cpu = false)(
    nd2x2x2x2,
    1128d
  ) { (m, doBackprop, cuda) =>
    Scope.root { implicit scope =>
      val input =
        param(STen.owned(NDArray.tensorFromNDArray(nd1x4x3x3, cuda)))
      val weight = param(STen.owned(NDArray.tensorFromNDArray(m, cuda)))

      val bias = param(lamp.saddle.fromVec(vec.ones(2), cuda))
      val output = new Convolution(
        scope = scope,
        input = input,
        weight = weight,
        bias = bias,
        stride = Array(1, 1),
        padding = Array(0, 0),
        dilation = Array(1, 1),
        transposed = false,
        outputPadding = Array(0, 0),
        groups = 2L
      ).value

      val L = output.sum
      if (doBackprop) {
        L.backprop()
      }
      (
        L.value.toMat.raw(0),
        weight.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
      )
    }
  }
  testGradientAndValueND("conv2d - wrt input")(nd1x2x3x3, 276d) {
    (m, doBackprop, cuda) =>
      Scope.root { implicit scope =>
        val input =
          param(STen.owned(NDArray.tensorFromNDArray(m, cuda)))
        val weight =
          param(STen.owned(NDArray.tensorFromNDArray(nd1x2x2x2, cuda)))

        val bias = param(lamp.saddle.fromVec(vec.ones(1), cuda))
        val output = new Convolution(
          scope = scope,
          input = input,
          weight = weight,
          bias = bias,
          stride = Array(1, 1),
          padding = Array(0, 0),
          dilation = Array(1, 1),
          transposed = false,
          outputPadding = Array(0, 0),
          groups = 1L
        ).value

        val L = output.sum
        if (doBackprop) {
          L.backprop()
        }
        (
          L.value.toMat.raw(0),
          input.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
        )
      }
  }
  testGradientAndValueND("conv2d - padded - wrt input")(nd1x2x3x3, 628d) {
    (m, doBackprop, cuda) =>
      Scope.root { implicit scope =>
        val input =
          param(STen.owned(NDArray.tensorFromNDArray(m, cuda)))
        val weight =
          param(STen.owned(NDArray.tensorFromNDArray(nd1x2x2x2, cuda)))

        val bias = param(lamp.saddle.fromVec(vec.ones(1), cuda))
        val output = new Convolution(
          scope = scope,
          input = input,
          weight = weight,
          bias = bias,
          stride = Array(1, 1),
          padding = Array(1, 1),
          dilation = Array(1, 1),
          transposed = false,
          outputPadding = Array(0, 0),
          groups = 1L
        ).value

        val L = output.sum
        if (doBackprop) {
          L.backprop()
        }
        (
          L.value.toMat.raw(0),
          input.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
        )
      }
  }
  testGradientAndValueND("conv2d - wrt bias")(ndx1, 276d) {
    (m, doBackprop, cuda) =>
      Scope.root { implicit scope =>
        val input =
          param(STen.owned(NDArray.tensorFromNDArray(nd1x2x3x3, cuda)))
        val weight =
          param(STen.owned(NDArray.tensorFromNDArray(nd1x2x2x2, cuda)))

        val bias = param(STen.owned(NDArray.tensorFromNDArray(m, cuda)))
        val output = new Convolution(
          scope = scope,
          input = input,
          weight = weight,
          bias = bias,
          stride = Array(1, 1),
          padding = Array(0, 0),
          dilation = Array(1, 1),
          transposed = false,
          outputPadding = Array(0, 0),
          groups = 1L
        ).value

        val L = output.sum
        if (doBackprop) {
          L.backprop()
        }
        (
          L.value.toMat.raw(0),
          bias.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
        )
      }
  }

  testGradientAndValueND("maxpool1d padded")(nd1x2x3, 32d) {
    (m, doBackprop, cuda) =>
      Scope.root { implicit scope =>
        val input =
          param(STen.owned(NDArray.tensorFromNDArray(m, cuda)))
        val output =
          new MaxPool1D(scope, input, 2, 1, 1, 1).value

        val L = output.sum
        if (doBackprop) {
          L.backprop()
        }
        (
          L.value.toMat.raw(0),
          input.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
        )
      }
  }
  testGradientAndValueND("maxpool1d unpadded")(nd1x2x3, 18d) {
    (m, doBackprop, cuda) =>
      Scope.root { implicit scope =>
        val input =
          param(STen.owned(NDArray.tensorFromNDArray(m, cuda)))

        val output =
          new MaxPool1D(scope, input, 2, 1, 0, 1).value

        val L = output.sum
        if (doBackprop) {
          L.backprop()
        }
        (
          L.value.toMat.raw(0),
          input.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
        )
      }
  }
  testGradientAndValueND("maxpool1d strided")(nd1x2x3, 7d) {
    (m, doBackprop, cuda) =>
      Scope.root { implicit scope =>
        val input =
          param(STen.owned(NDArray.tensorFromNDArray(m, cuda)))

        val output =
          new MaxPool1D(scope, input, 2, 2, 0, 1).value

        val L = output.sum
        if (doBackprop) {
          L.backprop()
        }
        (
          L.value.toMat.raw(0),
          input.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
        )
      }
  }
  testGradientAndValueND("maxpool2d strided")(nd1x2x3x3, 17d) {
    (m, doBackprop, cuda) =>
      Scope.root { implicit scope =>
        val input =
          param(STen.owned(NDArray.tensorFromNDArray(m, cuda)))

        val output =
          new MaxPool2D(scope, input, 2, 2, 0, 1).value

        val L = output.sum
        if (doBackprop) {
          L.backprop()
        }
        (
          L.value.toMat.raw(0),
          input.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
        )
      }
  }
  testGradientAndValueND("maxpool2d strided padded")(nd1x2x3x3, 68d) {
    (m, doBackprop, cuda) =>
      Scope.root { implicit scope =>
        val input =
          param(STen.owned(NDArray.tensorFromNDArray(m, cuda)))

        val output =
          new MaxPool2D(scope, input, 2, 2, 1, 1).value

        val L = output.sum
        if (doBackprop) {
          L.backprop()
        }
        (
          L.value.toMat.raw(0),
          input.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
        )
      }
  }
  testGradientAndValueND("avgpool2d strided padded")(nd1x2x3x3, 38.25) {
    (m, doBackprop, cuda) =>
      Scope.root { implicit scope =>
        val input =
          param(STen.owned(NDArray.tensorFromNDArray(m, cuda)))

        val output =
          new AvgPool2D(scope, input, 2, 2, 1).value

        val L = output.sum
        if (doBackprop) {
          L.backprop()
        }
        (
          L.value.toMat.raw(0),
          input.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
        )
      }
  }
  testGradientAndValue("batch norm 1d - wrt to input")(mat2x3, 0d) {
    (m, doBackprop, cuda) =>
      Scope.root { implicit scope =>
        val input =
          param(lamp.saddle.fromMat(m, cuda))
        val weight = param(lamp.saddle.fromVec(Vec(1d, 2d, 3d), cuda))

        val bias = param(lamp.saddle.fromVec(vec.zeros(3), cuda))
        val runningMean = lamp.saddle.fromVec(vec.ones(3), cuda)
        val runningVar = lamp.saddle.fromVec(vec.ones(3), cuda)

        val output =
          new BatchNorm(
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
          L.value.toMat.raw(0),
          input.partialDerivative.map(t => t.toMat)
        )
      }
  }
  testGradientAndValueND("batch norm 1d - wrt to weight")(ndx3, 0d) {
    (m, doBackprop, cuda) =>
      Scope.root { implicit scope =>
        val input =
          param(lamp.saddle.fromMat(mat2x3, cuda))
        val weight = param(STen.owned(NDArray.tensorFromNDArray(m, cuda)))

        val bias = param(lamp.saddle.fromVec(vec.zeros(3), cuda))
        val runningMean = lamp.saddle.fromVec(vec.ones(3), cuda)
        val runningVar = lamp.saddle.fromVec(vec.ones(3), cuda)

        val output =
          new BatchNorm(
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
          L.value.toMat.raw(0),
          weight.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
        )
      }
  }
  testGradientAndValueND("batch norm 1d - wrt to bias")(ndx3, 12d) {
    (m, doBackprop, cuda) =>
      Scope.root { implicit scope =>
        val input =
          param(lamp.saddle.fromMat(mat2x3, cuda))
        val bias = param(STen.owned(NDArray.tensorFromNDArray(m, cuda)))

        val weight = param(lamp.saddle.fromVec(vec.zeros(3), cuda))
        val runningMean = lamp.saddle.fromVec(vec.ones(3), cuda)
        val runningVar = lamp.saddle.fromVec(vec.ones(3), cuda)

        val output =
          new BatchNorm(
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
          L.value.toMat.raw(0),
          bias.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
        )
      }
  }

  testGradientAndValue("layer norm 1d - wrt to input")(mat2x3, 0.8165) {
    (m, doBackprop, cuda) =>
      Scope.root { implicit scope =>
        val input =
          param(lamp.saddle.fromMat(m, cuda))
        val weight = param(lamp.saddle.fromVec(Vec(1d, 2d, 3d), cuda))

        val bias = param(lamp.saddle.fromVec(vec.zeros(3), cuda))

        val output =
          new LayerNormOp(
            scope,
            input,
            Option(weight),
            Option(bias),
            List(3L),
            eps = 1e-5
          ).value.mean(List(0, 1))

        val L = output
        if (doBackprop) {
          L.backprop()
        }
        (
          L.value.toMat.raw(0),
          input.partialDerivative.map(t => t.toMat)
        )
      }
  }
  testGradientAndValue("layer norm 1d - wrt to input - no scale and no bias")(mat2x3, 0.0) {
    (m, doBackprop, cuda) =>
      Scope.root { implicit scope =>
        val input =
          param(lamp.saddle.fromMat(m, cuda))
        val output =
          new LayerNormOp(
            scope,
            input,
            None,
            None,
            List(3L),
            eps = 1e-5
          ).value.mean(List(0, 1))

        val L = output
        if (doBackprop) {
          L.backprop()
        }
        (
          L.value.toMat.raw(0),
          input.partialDerivative.map(t => t.toMat)
        )
      }
  }
  testGradientAndValueND("layer norm 1d - wrt to weight")(ndx3, 0.8165) {
    (m, doBackprop, cuda) =>
      Scope.root { implicit scope =>
        val input =
          param(lamp.saddle.fromMat(mat2x3, cuda))
        val weight = param(STen.owned(NDArray.tensorFromNDArray(m, cuda)))

        val bias = param(lamp.saddle.fromVec(vec.zeros(3), cuda))

        val output =
          new LayerNormOp(
            scope,
            input,
            Option(weight),
            Option(bias),
            List(3L),
            eps = 1e-5
          ).value

        val L = output.mean(List(0, 1))
        if (doBackprop) {
          L.backprop()
        }
        (
          L.value.toMat.raw(0),
          weight.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
        )
      }
  }
  testGradientAndValueND("layer norm 1d - wrt to weight - no bias")(ndx3, 0.8165) {
    (m, doBackprop, cuda) =>
      Scope.root { implicit scope =>
        val input =
          param(lamp.saddle.fromMat(mat2x3, cuda))
        val weight = param(STen.owned(NDArray.tensorFromNDArray(m, cuda)))


        val output =
          new LayerNormOp(
            scope,
            input,
            Option(weight),
            None,
            List(3L),
            eps = 1e-5
          ).value

        val L = output.mean(List(0, 1))
        if (doBackprop) {
          L.backprop()
        }
        (
          L.value.toMat.raw(0),
          weight.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
        )
      }
  }
  testGradientAndValueND("layer norm 1d - wrt to bias")(ndx3, 2d) {
    (m, doBackprop, cuda) =>
      Scope.root { implicit scope =>
        val input =
          param(lamp.saddle.fromMat(mat2x3, cuda))
        val bias = param(STen.owned(NDArray.tensorFromNDArray(m, cuda)))

        val weight = param(lamp.saddle.fromVec(vec.zeros(3), cuda))

        val output =
          new LayerNormOp(
            scope,
            input,
            Option(weight),
            Option(bias),
            List(3L),
            eps = 1e-5
          ).value

        val L = output.mean(List(0, 1))
        if (doBackprop) {
          L.backprop()
        }
        (
          L.value.toMat.raw(0),
          bias.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
        )
      }
  }
  testGradientAndValueND("layer norm 1d - wrt to bias - no scale")(ndx3, 2d) {
    (m, doBackprop, cuda) =>
      Scope.root { implicit scope =>
        val input =
          param(lamp.saddle.fromMat(mat2x3, cuda))
        val bias = param(STen.owned(NDArray.tensorFromNDArray(m, cuda)))


        val output =
          new LayerNormOp(
            scope,
            input,
            None,
            Option(bias),
            List(3L),
            eps = 1e-5
          ).value

        val L = output.mean(List(0, 1))
        if (doBackprop) {
          L.backprop()
        }
        (
          L.value.toMat.raw(0),
          bias.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
        )
      }
  }
  testGradientAndValueND("bmm - wrt left")(nd3x2x3, 489d) {
    (m, doBackprop, cuda) =>
      Scope.root { implicit scope =>
        val input =
          param(STen.owned(NDArray.tensorFromNDArray(m, cuda)))
        val other = param(STen.owned(NDArray.tensorFromNDArray(nd3x3x2, cuda)))
        val output = input.bmm(other)

        val L = output.sum
        if (doBackprop) {
          L.backprop()
        }
        (
          L.value.toMat.raw(0),
          input.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
        )
      }
  }
  testGradientAndValueND("bmm - wrt right")(nd3x3x2, 489d) {
    (m, doBackprop, cuda) =>
      Scope.root { implicit scope =>
        val input =
          param(STen.owned(NDArray.tensorFromNDArray(m, cuda)))
        val other = param(STen.owned(NDArray.tensorFromNDArray(nd3x2x3, cuda)))
        val output = other.bmm(input)

        val L = output.sum
        if (doBackprop) {
          L.backprop()
        }
        (
          L.value.toMat.raw(0),
          input.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
        )
      }
  }
  testGradientAndValueND("batch norm 2d - wrt to input")(nd1x2x3, 6d) {
    (m, doBackprop, cuda) =>
      Scope.root { implicit scope =>
        val input =
          param(STen.owned(NDArray.tensorFromNDArray(m, cuda)))
        val bias = param(lamp.saddle.fromVec(vec.ones(6), cuda))

        val weight = param(lamp.saddle.fromVec(vec.ones(6), cuda))
        val runningMean = lamp.saddle.fromVec(vec.ones(6), cuda)
        val runningVar = lamp.saddle.fromVec(vec.ones(6), cuda)

        val output =
          new BatchNorm(
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
          L.value.toMat.raw(0),
          input.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
        )
      }
  }
  testGradientAndValueND("batch norm 2d - wrt to weights")(ndx6, 6d) {
    (m, doBackprop, cuda) =>
      Scope.root { implicit scope =>
        val input =
          param(STen.owned(NDArray.tensorFromNDArray(nd1x2x3, cuda)))
        val bias = param(lamp.saddle.fromVec(vec.ones(6), cuda))

        val weight = param(STen.owned(NDArray.tensorFromNDArray(m, cuda)))
        val runningMean = lamp.saddle.fromVec(vec.ones(6), cuda)
        val runningVar = lamp.saddle.fromVec(vec.ones(6), cuda)

        val output =
          new BatchNorm(
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
          L.value.toMat.raw(0),
          weight.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
        )
      }
  }
  testGradientAndValueND("batch norm 2d - wrt to bias")(ndx6, 21d) {
    (m, doBackprop, cuda) =>
      Scope.root { implicit scope =>
        val input =
          param(STen.owned(NDArray.tensorFromNDArray(nd1x2x3, cuda)))
        val weight = param(lamp.saddle.fromVec(vec.ones(6), cuda))

        val bias = param(STen.owned(NDArray.tensorFromNDArray(m, cuda)))
        val runningMean = lamp.saddle.fromVec(vec.ones(6), cuda)
        val runningVar = lamp.saddle.fromVec(vec.ones(6), cuda)

        val output =
          new BatchNorm(
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
          L.value.toMat.raw(0),
          bias.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
        )
      }
  }
  testGradientAndValueND("batch norm 3d - wrt to input")(nd1x2x3x3, 18d) {
    (m, doBackprop, cuda) =>
      Scope.root { implicit scope =>
        val input =
          param(STen.owned(NDArray.tensorFromNDArray(m, cuda)))
        val bias = param(lamp.saddle.fromVec(vec.ones(18), cuda))

        val weight = param(lamp.saddle.fromVec(vec.ones(18), cuda))
        val runningMean = lamp.saddle.fromVec(vec.ones(18), cuda)
        val runningVar = lamp.saddle.fromVec(vec.ones(18), cuda)

        val output =
          new BatchNorm(
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
          L.value.toMat.raw(0),
          input.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
        )
      }
  }
  testGradientAndValueND("batch norm 3d - wrt to weights")(ndx18, 18d) {
    (m, doBackprop, cuda) =>
      Scope.root { implicit scope =>
        val input =
          param(STen.owned(NDArray.tensorFromNDArray(nd1x2x3x3, cuda)))
        val bias = param(lamp.saddle.fromVec(vec.ones(18), cuda))

        val weight = param(STen.owned(NDArray.tensorFromNDArray(m, cuda)))
        val runningMean = lamp.saddle.fromVec(vec.ones(18), cuda)
        val runningVar = lamp.saddle.fromVec(vec.ones(18), cuda)

        val output =
          new BatchNorm(
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
          L.value.toMat.raw(0),
          weight.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
        )
      }
  }
  testGradientAndValueND("batch norm 3d - wrt to bias")(ndx18, 63d) {
    (m, doBackprop, cuda) =>
      Scope.root { implicit scope =>
        val input =
          param(STen.owned(NDArray.tensorFromNDArray(nd1x2x3x3, cuda)))
        val weights = param(lamp.saddle.fromVec(vec.ones(18), cuda))

        val bias = param(STen.owned(NDArray.tensorFromNDArray(m, cuda)))
        val runningMean = lamp.saddle.fromVec(vec.ones(18), cuda)
        val runningVar = lamp.saddle.fromVec(vec.ones(18), cuda)

        val output =
          new BatchNorm(
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
          L.value.toMat.raw(0),
          bias.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
        )
      }
  }
  testGradientAndValueND("BatchNorm2D - wrt to input")(nd1x2x3x3, 18d) {
    (m, doBackprop, cuda) =>
      Scope.root { implicit scope =>
        val input =
          param(STen.owned(NDArray.tensorFromNDArray(m, cuda)))
        val weights = param(lamp.saddle.fromVec(vec.ones(2), cuda))

        val bias = param(STen.owned(NDArray.tensorFromNDArray(ndx2, cuda)))
        val runningMean = lamp.saddle.fromVec(vec.zeros(2), cuda)
        val runningVar = lamp.saddle.fromVec(vec.zeros(2), cuda)

        val output =
          new BatchNorm2D(
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
          L.value.toMat.raw(0),
          input.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
        )
      }
  }
  testGradientAndValueND("BatchNorm2D - wrt to weights")(ndx2, 18d) {
    (m, doBackprop, cuda) =>
      Scope.root { implicit scope =>
        val input =
          param(STen.owned(NDArray.tensorFromNDArray(nd1x2x3x3, cuda)))
        val weights = param(STen.owned(NDArray.tensorFromNDArray(m, cuda)))

        val bias = param(STen.owned(NDArray.tensorFromNDArray(ndx2, cuda)))
        val runningMean = lamp.saddle.fromVec(vec.zeros(2), cuda)
        val runningVar = lamp.saddle.fromVec(vec.zeros(2), cuda)

        val output =
          new BatchNorm2D(
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
          L.value.toMat.raw(0),
          weights.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
        )
      }
  }
  testGradientAndValueND("BatchNorm2D - wrt to bias")(ndx2, 18d) {
    (m, doBackprop, cuda) =>
      Scope.root { implicit scope =>
        val input =
          param(STen.owned(NDArray.tensorFromNDArray(nd1x2x3x3, cuda)))
        val bias = param(STen.owned(NDArray.tensorFromNDArray(m, cuda)))

        val weights = param(STen.owned(NDArray.tensorFromNDArray(ndx2, cuda)))
        val runningMean = lamp.saddle.fromVec(vec.zeros(2), cuda)
        val runningVar = lamp.saddle.fromVec(vec.zeros(2), cuda)

        val output =
          new BatchNorm2D(
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
          L.value.toMat.raw(0),
          bias.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
        )
      }
  }
  testGradientAndValueND("flatten ")(nd1x2x3x3, 153d) { (m, doBackprop, cuda) =>
    Scope.root { implicit scope =>
      val input =
        param(STen.owned(NDArray.tensorFromNDArray(m, cuda)))

      val output = input.flattenLastDimensions(3)(scope)

      assert(output.shape == List(1, 18))

      val L = output.sum
      if (doBackprop) {
        L.backprop()
      }
      (
        L.value.toMat.raw(0),
        input.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
      )
    }
  }
  testGradientAndValueND("select 0 0 ")(nd1x2x3x3, 153d) {
    (m, doBackprop, cuda) =>
      Scope.root { implicit scope =>
        val input =
          param(STen.owned(NDArray.tensorFromNDArray(m, cuda)))

        val output =
          input.select(0, 0)

        assert(output.shape == List(2, 3, 3))

        val L = output.sum
        if (doBackprop) {
          L.backprop()
        }
        (
          L.value.toMat.raw(0),
          input.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
        )
      }
  }
  testGradientAndValueND("select 2 1 ")(nd1x2x3x3, 51d) {
    (m, doBackprop, cuda) =>
      Scope.root { implicit scope =>
        val input =
          param(STen.owned(NDArray.tensorFromNDArray(m, cuda)))

        val output =
          input.select(2, 1)

        assert(output.shape == List(1, 2, 3))

        val L = output.sum
        if (doBackprop) {
          L.backprop()
        }
        (
          L.value.toMat.raw(0),
          input.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
        )
      }
  }
  testGradientAndValueND("slice ")(nd1x2x3x3, 120d) { (m, doBackprop, cuda) =>
    Scope.root { implicit scope =>
      val input =
        param(STen.owned(NDArray.tensorFromNDArray(m, cuda)))

      val output =
        input.slice(dim = 2, start = 1, end = 3, step = 1)

      assert(output.shape == List(1, 2, 2, 3))

      val L = output.sum
      if (doBackprop) {
        L.backprop()
      }
      (
        L.value.toMat.raw(0),
        input.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
      )
    }
  }

  testGradientAndValueND("stack 0")(nd1x2x3, 42d) { (m, doBackprop, cuda) =>
    Scope.root { implicit scope =>
      val input =
        param(STen.owned(NDArray.tensorFromNDArray(m, cuda)))

      val output =
        new Stack(scope, List(input, input), 0).value

      assert(output.shape == List(2, 1, 2, 3))

      val L = output.sum
      if (doBackprop) {
        L.backprop()
      }
      (
        L.value.toMat.raw(0),
        input.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
      )
    }
  }
  testGradientAndValueND("stack 1")(nd1x2x3, 42d) { (m, doBackprop, cuda) =>
    Scope.root { implicit scope =>
      val input =
        param(STen.owned(NDArray.tensorFromNDArray(m, cuda)))

      val output =
        new Stack(scope, List(input, input), 1).value

      assert(output.shape == List(1, 2, 2, 3))

      val L = output.sum
      if (doBackprop) {
        L.backprop()
      }
      (
        L.value.toMat.raw(0),
        input.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
      )
    }
  }
  testGradientAndValueND("cat 1 ")(nd1x2x3, 84d) { (m, doBackprop, cuda) =>
    Scope.root { implicit scope =>
      val input =
        param(STen.owned(NDArray.tensorFromNDArray(m, cuda)))
      val t2 =
        param(STen.owned(NDArray.tensorFromNDArray(nd1x2x3_2, cuda)))

      val output =
        new Concatenate(scope, List(input, t2), 1).value

      assert(output.shape == List(1, 4, 3))

      val L = output.sum
      if (doBackprop) {
        L.backprop()
      }
      (
        L.value.toMat.raw(0),
        input.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
      )
    }
  }
  testGradientAndValueND("cat 2 ")(nd1x2x3, 84d) { (m, doBackprop, cuda) =>
    Scope.root { implicit scope =>
      val input =
        param(STen.owned(NDArray.tensorFromNDArray(m, cuda)))
      val t2 =
        param(STen.owned(NDArray.tensorFromNDArray(nd1x2x3_2, cuda)))
      val output =
        new Concatenate(scope, List(input, t2), 2).value

      assert(output.shape == List(1, 2, 6))

      val L = output.sum
      if (doBackprop) {
        L.backprop()
      }
      (
        L.value.toMat.raw(0),
        input.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
      )
    }
  }

  testGradientAndValueND("view 1 ")(nd1x2x3, 21d) { (m, doBackprop, cuda) =>
    Scope.root { implicit scope =>
      val input =
        param(STen.owned(NDArray.tensorFromNDArray(m, cuda)))

      val output = input.view(List(1, 1, 2, 3))

      assert(output.shape == List(1, 1, 2, 3))

      val L = output.sum
      if (doBackprop) {
        L.backprop()
      }
      (
        L.value.toMat.raw(0),
        input.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
      )
    }
  }
  testGradientAndValueND("reshape 1 ")(nd1x2x3, 21d) { (m, doBackprop, cuda) =>
    Scope.root { implicit scope =>
      val input =
        param(STen.owned(NDArray.tensorFromNDArray(m, cuda)))

      val output = input.reshape(List(1, 1, 2, 3))

      assert(output.shape == List(1, 1, 2, 3))

      val L = output.sum
      if (doBackprop) {
        L.backprop()
      }
      (
        L.value.toMat.raw(0),
        input.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
      )
    }
  }
  testGradientAndValue("embedding ")(mat2x3, 240d) { (m, doBackprop, cuda) =>
    Scope.root { implicit scope =>
      val weight =
        param(lamp.saddle.fromMat(m, cuda))
      val input =
        param(lamp.saddle.fromLongMat(mat.ones(4, 5).map(_.toLong), cuda))

      val output = new Embedding(scope, input, weight).value

      assert(output.shape == List(4, 5, 3))

      val L = output.sum
      if (doBackprop) {
        L.backprop()
      }
      (
        L.value.toMat.raw(0),
        weight.partialDerivative.map(t => t.toMat)
      )
    }
  }

  testGradientAndValueND("tranposed conv2d - wrt input")(nd1x2x3x3, 628d) {
    (m, doBackprop, cuda) =>
      Scope.root { implicit scope =>
        val input =
          param(STen.owned(NDArray.tensorFromNDArray(m, cuda)))
        val weight =
          param(
            STen
              .owned(NDArray.tensorFromNDArray(nd1x2x2x2, cuda))
              .transpose(0, 1)
          )

        val bias = param(lamp.saddle.fromVec(vec.ones(1), cuda))
        val output = new Convolution(
          scope = scope,
          input = input,
          weight = weight,
          bias = bias,
          stride = Array(1, 1),
          padding = Array(0, 0),
          dilation = Array(1, 1),
          transposed = true,
          outputPadding = Array(0, 0),
          groups = 1L
        ).value

        assert(output.shape == List(1, 1, 4, 4))
        val L = output.sum
        if (doBackprop) {
          L.backprop()
        }
        (
          L.value.toMat.raw(0),
          input.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
        )
      }
  }
  testGradientAndValueND("tranposed conv2d - wrt input - padded")(
    nd1x2x3x3,
    276d
  ) { (m, doBackprop, cuda) =>
    Scope.root { implicit scope =>
      val input =
        param(STen.owned(NDArray.tensorFromNDArray(m, cuda)))
      val weight =
        param(
          STen
            .owned(NDArray.tensorFromNDArray(nd1x2x2x2, cuda))
            .transpose(0, 1)
        )

      val bias = param(lamp.saddle.fromVec(vec.ones(1), cuda))
      val output = new Convolution(
        scope = scope,
        input = input,
        weight = weight,
        bias = bias,
        stride = Array(1, 1),
        padding = Array(1, 1),
        dilation = Array(1, 1),
        transposed = true,
        outputPadding = Array(0, 0),
        groups = 1L
      ).value

      assert(output.shape == List(1, 1, 2, 2))
      val L = output.sum
      if (doBackprop) {
        L.backprop()
      }
      (
        L.value.toMat.raw(0),
        input.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
      )
    }
  }
  testGradientAndValueND("tranposed conv2d - wrt weight")(nd1x2x2x2, 628d) {
    (m, doBackprop, cuda) =>
      Scope.root { implicit scope =>
        val input =
          param(STen.owned(NDArray.tensorFromNDArray(nd1x2x3x3, cuda)))
        val weight =
          param(
            STen
              .owned(NDArray.tensorFromNDArray(m, cuda))
              .transpose(0, 1)
          )

        val bias = param(lamp.saddle.fromVec(vec.ones(1), cuda))
        val output = new Convolution(
          scope = scope,
          input = input,
          weight = weight,
          bias = bias,
          stride = Array(1, 1),
          padding = Array(0, 0),
          dilation = Array(1, 1),
          transposed = true,
          outputPadding = Array(0, 0),
          groups = 1L
        ).value

        assert(output.shape == List(1, 1, 4, 4))
        val L = output.sum
        if (doBackprop) {
          L.backprop()
        }
        (
          L.value.toMat.raw(0),
          weight.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
        )
      }
  }
  testGradientAndValueND("tranposed conv2d - wrt bias")(ndx1, 628d) {
    (m, doBackprop, cuda) =>
      Scope.root { implicit scope =>
        val input =
          param(STen.owned(NDArray.tensorFromNDArray(nd1x2x3x3, cuda)))
        val weight =
          param(
            STen
              .owned(NDArray.tensorFromNDArray(nd1x2x2x2, cuda))
              .transpose(0, 1)
          )

        val bias = param(STen.owned(NDArray.tensorFromNDArray(m, cuda)))
        val output = new Convolution(
          scope = scope,
          input = input,
          weight = weight,
          bias = bias,
          stride = Array(1, 1),
          padding = Array(0, 0),
          dilation = Array(1, 1),
          transposed = true,
          outputPadding = Array(0, 0),
          groups = 1L
        ).value

        assert(output.shape == List(1, 1, 4, 4))
        val L = output.sum
        if (doBackprop) {
          L.backprop()
        }
        (
          L.value.toMat.raw(0),
          bias.partialDerivative.map(t => NDArray.tensorToNDArray(t.value))
        )
      }
  }

}
