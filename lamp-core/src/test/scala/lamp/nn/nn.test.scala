package lamp.nn

import org.saddle._
import org.saddle.ops.BinOps._
import org.saddle.linalg._
import org.scalatest.funsuite.AnyFunSuite
import aten.ATen
import lamp.autograd._
import aten.TensorOptions
import org.scalatest.Tag
import lamp.syntax
import lamp.util.NDArray
import aten.Tensor
import cats.effect.IO
import cats.effect.concurrent.Ref

object CudaTest extends Tag("cuda")

class NNSuite extends AnyFunSuite {
  def test1(id: String)(fun: Boolean => Unit) = {
    test(id) { fun(false) }
    test(id + "/CUDA", CudaTest) { fun(true) }
  }
  def testGradientAndValue(
      id: String,
      cuda: Boolean = false
  )(m: Mat[Double], moduleF: () => Module, expectedValue: Double) =
    test(id + ": gradient is correct", (if (cuda) List(CudaTest) else Nil): _*) {
      val d = const(TensorHelpers.fromMat(m, cuda))
      val module = moduleF()

      {
        val module1 = moduleF()
        val state = module1.state
        val modifiedState = state.map {
          case (v, ptag) =>
            ATen.mul_1(v.value, -1d)
        }
        val module2 = module1.load(modifiedState)
        (module2.state zip modifiedState).foreach {
          case ((st1, _), (st2)) =>
            assert(
              NDArray.tensorToNDArray(st1.value).toVec == NDArray
                .tensorToNDArray(st2)
                .toVec
            )
        }
      }

      val output = module.forward(d)
      val sum = output.sum
      val value = TensorHelpers.toMat(sum.value).raw(0)
      val gradAuto = module.gradients(sum).map(_.get).map(TensorHelpers.toMat)
      val gradNum = module.parameters.map {
        case (paramT, _) =>
          val oldparam = ATen.clone(paramT.value)
          val param = TensorHelpers.toMat(paramT.value)
          def f(p: Mat[Double]) = {
            val p2 = TensorHelpers.fromMat(p, cuda)
            ATen.zero_(paramT.value)
            ATen.add_out(
              paramT.value,
              paramT.value,
              ATen._unsafe_view(p2, paramT.sizes.toArray),
              1d
            )
            TensorHelpers.toMat(module.forward(d).sum.value).raw(0)
          }
          val eps = 1e-6
          val r = mat.zeros(param.numRows, param.numCols).mapRows {
            case (row, i) =>
              (0 until row.length).map { j =>
                val epsM = mat.zeros(param.numRows, param.numCols)
                epsM(i, j) = eps

                (f(param + epsM) - f(param - epsM)) / (2 * eps)
              }.toVec
          }
          ATen.zero_(paramT.value)
          ATen.add_out(paramT.value, paramT.value, oldparam, 1d)
          r
      }
      assert(gradAuto.size == gradNum.size)
      gradAuto.zip(gradNum).foreach {
        case (a, b) =>
          assert(a.roundTo(4) == b.roundTo(4))
      }
      assert(value == expectedValue)

    }
  def testGradientAndValueND[T](
      id: String,
      st: T,
      cuda: Boolean = false
  )(
      m: NDArray[Double],
      moduleF: () => StatefulModule[T],
      expectedValue: Double
  ) =
    test(id + ": gradient is correct", (if (cuda) List(CudaTest) else Nil): _*) {
      val d = const(NDArray.tensorFromNDArray(m, cuda))
      val module = moduleF()

      {
        val module1 = moduleF()
        val state = module1.state
        val modifiedState = state.map {
          case (v, ptag) =>
            ATen.mul_1(v.value, -1d)
        }
        val module2 = module1.load(modifiedState)
        (module2.state zip modifiedState).foreach {
          case ((st1, _), (st2)) =>
            assert(
              NDArray.tensorToNDArray(st1.value).toVec == NDArray
                .tensorToNDArray(st2)
                .toVec
            )
        }
      }

      val output = module.forward1(d, st)._1
      val sum = output.sum
      val value = NDArray.tensorToNDArray(sum.value).data(0)
      val gradAuto =
        module.gradients(sum).map(_.get).map(NDArray.tensorToNDArray)
      val gradNum = module.parameters.map {
        case (paramT, _) =>
          val oldparam = ATen.clone(paramT.value)
          val param = NDArray.tensorToNDArray(paramT.value)
          def f(p: NDArray[Double]) = {
            val p2 = NDArray.tensorFromNDArray(p, cuda)
            ATen.zero_(paramT.value)
            ATen.add_out(
              paramT.value,
              paramT.value,
              p2,
              1d
            )
            TensorHelpers.toMat(module.forward1(d, st)._1.sum.value).raw(0)
          }
          val eps = 1e-6
          val r = NDArray.zeros(paramT.shape.map(_.toInt)).mapWithIndex {
            case (_, idx) =>
              val epsM = NDArray.zeros(paramT.shape.map(_.toInt))
              epsM.set(idx, eps)
              val a = f(param + epsM)
              val b = f(param - epsM)
              val r = (a - b) / (2 * eps)
              r
          }
          ATen.zero_(paramT.value)
          ATen.add_out(paramT.value, paramT.value, oldparam, 1d)
          r
      }
      assert(gradAuto.size == gradNum.size)
      gradAuto.zip(gradNum).foreach {
        case (a, b) =>
          assert(a.toVec.roundTo(4) == b.toVec.roundTo(4))
      }
      assert(Vec(value).roundTo(4) == Vec(expectedValue).roundTo(4))

    }
  def testGradientAndValueNDLong[T](
      id: String,
      st: T,
      cuda: Boolean = false
  )(
      m: NDArray[Long],
      moduleF: () => StatefulModule[T],
      expectedValue: Double
  ) =
    test(id + ": gradient is correct", (if (cuda) List(CudaTest) else Nil): _*) {
      val d = const(NDArray.tensorFromLongNDArray(m, cuda))
      val module = moduleF()

      {
        val module1 = moduleF()
        val state = module1.state
        val modifiedState = state.map {
          case (v, ptag) =>
            ATen.mul_1(v.value, -1d)
        }
        val module2 = module1.load(modifiedState)
        (module2.state zip modifiedState).foreach {
          case ((st1, _), (st2)) =>
            assert(
              NDArray.tensorToNDArray(st1.value).toVec == NDArray
                .tensorToNDArray(st2)
                .toVec
            )
        }
      }

      val output = module.forward1(d, st)._1
      val sum = output.sum
      val value = NDArray.tensorToNDArray(sum.value).data(0)
      val gradAuto =
        module.gradients(sum).map(_.get).map(NDArray.tensorToNDArray)
      val gradNum = module.parameters.map {
        case (paramT, _) =>
          val oldparam = ATen.clone(paramT.value)
          val param = NDArray.tensorToNDArray(paramT.value)
          def f(p: NDArray[Double]) = {
            val p2 = NDArray.tensorFromNDArray(p, cuda)
            ATen.zero_(paramT.value)
            ATen.add_out(
              paramT.value,
              paramT.value,
              p2,
              1d
            )
            TensorHelpers.toMat(module.forward1(d, st)._1.sum.value).raw(0)
          }
          val eps = 1e-6
          val r = NDArray.zeros(paramT.shape.map(_.toInt)).mapWithIndex {
            case (_, idx) =>
              val epsM = NDArray.zeros(paramT.shape.map(_.toInt))
              epsM.set(idx, eps)
              val a = f(param + epsM)
              val b = f(param - epsM)
              val r = (a - b) / (2 * eps)
              r
          }
          ATen.zero_(paramT.value)
          ATen.add_out(paramT.value, paramT.value, oldparam, 1d)
          r
      }
      assert(gradAuto.size == gradNum.size)
      gradAuto.zip(gradNum).foreach {
        case (a, b) =>
          assert(a.toVec.roundTo(4) == b.toVec.roundTo(4))
      }
      assert(Vec(value).roundTo(4) == Vec(expectedValue).roundTo(4))

    }

  val mat2x3 = Mat(Vec(1d, 2d), Vec(3d, 4d), Vec(5d, 6d))
  val mat3x2 = mat2x3.T
  val nd2x3L = NDArray(mat2x3.map(_ => 1L).toArray, List(2, 3))
  val nd1x2x3 = NDArray(mat2x3.toArray, List(1, 2, 3))
  val nd1x2x3x3 =
    NDArray((0 until 18).toArray.map(_.toDouble), List(1, 2, 3, 3))
  val nd2x3x2 =
    NDArray((0 until 12).toArray.map(_.toDouble), List(2, 3, 2))

  test("linear") {
    val linear = Linear(3, 1, TensorOptions.dtypeDouble())
    val output = linear.forward(const(TensorHelpers.fromMat(mat2x3)))
    val sum = output.sum
    assert(output.value.sizes.toList == List(2, 1))
    val grad = linear.gradients(sum)
    assert(grad.size == 2)

  }
  testGradientAndValue("Linear 0")(
    mat2x3,
    () =>
      Linear(
        param(ATen.ones(Array(1, 3), TensorOptions.dtypeDouble)),
        Some(param(ATen.ones(Array(1), TensorOptions.dtypeDouble)))
      ),
    23d
  )

  testGradientAndValue("WeightNormLinear 0")(
    mat2x3,
    () =>
      WeightNormLinear(
        param(ATen.ones(Array(1, 3), TensorOptions.dtypeDouble)),
        param(ATen.ones(Array(1, 3), TensorOptions.dtypeDouble)),
        Some(param(ATen.ones(Array(1), TensorOptions.dtypeDouble)))
      ),
    23d
  )
  testGradientAndValue("Logistic 1")(
    mat3x2,
    () => LogisticRegression1(2, 3, const(TensorHelpers.fromMat(mat.ident(3)))),
    151.0000008318073
  )
  testGradientAndValue("Logistic 2")(
    mat3x2,
    () => LogisticRegression2(2, 3, const(TensorHelpers.fromMat(mat.ident(3)))),
    12.295836866004327
  )
  testGradientAndValue("Logistic 2 - cuda", true)(
    mat3x2,
    () =>
      LogisticRegression2(
        2,
        3,
        const(TensorHelpers.fromMat(mat.ident(3), cuda = true))
      ),
    12.295836866004326
  )
  testGradientAndValue("Mlp1 ", false)(
    mat3x2,
    () =>
      Mlp1(
        2,
        3,
        const(TensorHelpers.fromMat(mat.ident(3), cuda = false))
      ),
    192.08796576929555
  )
  testGradientAndValue("Mlp1 - cuda", true)(
    mat3x2,
    () =>
      Mlp1(
        2,
        3,
        const(TensorHelpers.fromMat(mat.ident(3), cuda = true))
      ),
    192.08796576929555
  )

  testGradientAndValueND("Conv1D ", (), false)(
    nd1x2x3,
    () =>
      Conv1D(
        param(ATen.ones(Array(1, 2, 3), TensorOptions.dtypeDouble)),
        param(ATen.ones(Array(1), TensorOptions.dtypeDouble)),
        stride = 1,
        padding = 0,
        dilation = 1,
        groups = 1
      ),
    22d
  )
  testGradientAndValueND("Conv1D/cuda ", (), true)(
    nd1x2x3,
    () =>
      Conv1D(
        param(ATen.ones(Array(1, 2, 3), TensorOptions.dtypeDouble.cuda)),
        param(ATen.ones(Array(1), TensorOptions.dtypeDouble.cuda)),
        stride = 1,
        padding = 0,
        dilation = 1,
        groups = 1
      ),
    22d
  )
  testGradientAndValueND("Conv2D ", (), false)(
    nd1x2x3x3,
    () =>
      Conv2D(
        param(ATen.ones(Array(1, 2, 3, 3), TensorOptions.dtypeDouble)),
        param(ATen.ones(Array(1), TensorOptions.dtypeDouble)),
        stride = 1,
        padding = 0,
        dilation = 1,
        groups = 1
      ),
    154d
  )
  testGradientAndValueND("Conv2D/cuda ", (), true)(
    nd1x2x3x3,
    () =>
      Conv2D(
        param(ATen.ones(Array(1, 2, 3, 3), TensorOptions.dtypeDouble.cuda)),
        param(ATen.ones(Array(1), TensorOptions.dtypeDouble.cuda)),
        stride = 1,
        padding = 0,
        dilation = 1,
        groups = 1
      ),
    154d
  )
  testGradientAndValueND("BatchNorm ", (), false)(
    nd1x2x3x3,
    () =>
      BatchNorm(
        18,
        TensorOptions.dtypeDouble()
      ),
    0d
  )

  testGradientAndValueNDLong("Embedding ", (), false)(
    nd2x3L,
    () =>
      Embedding(
        weights = param(ATen.ones(Array(2, 4), TensorOptions.dtypeDouble))
      ),
    24d
  )
  testGradientAndValueND("RNN ", Option.empty[Variable], false)(
    nd2x3x2,
    () =>
      RNN(
        weightXh = param(ATen.ones(Array(2, 4), TensorOptions.dtypeDouble)),
        weightHh = param(ATen.ones(Array(4, 4), TensorOptions.dtypeDouble)),
        weightHq = param(ATen.ones(Array(4, 3), TensorOptions.dtypeDouble)),
        biasQ = param(ATen.ones(Array(3), TensorOptions.dtypeDouble)),
        biasH = param(ATen.ones(Array(4), TensorOptions.dtypeDouble)),
        dropout = 0d,
        train = true
      ),
    89.5682
  )
  testGradientAndValueND("GRU ", Option.empty[Variable], false)(
    nd2x3x2,
    () =>
      GRU(
        weightXh = param(ATen.ones(Array(2, 4), TensorOptions.dtypeDouble)),
        weightXr = param(ATen.ones(Array(2, 4), TensorOptions.dtypeDouble)),
        weightXz = param(ATen.ones(Array(2, 4), TensorOptions.dtypeDouble)),
        weightHh = param(ATen.ones(Array(4, 4), TensorOptions.dtypeDouble)),
        weightHz = param(ATen.ones(Array(4, 4), TensorOptions.dtypeDouble)),
        weightHr = param(ATen.ones(Array(4, 4), TensorOptions.dtypeDouble)),
        weightHq = param(ATen.ones(Array(4, 3), TensorOptions.dtypeDouble)),
        biasQ = param(ATen.ones(Array(3), TensorOptions.dtypeDouble)),
        biasH = param(ATen.ones(Array(4), TensorOptions.dtypeDouble)),
        biasZ = param(ATen.ones(Array(4), TensorOptions.dtypeDouble)),
        biasR = param(ATen.ones(Array(4), TensorOptions.dtypeDouble)),
        dropout = 0d,
        train = true
      ),
    20.8184
  )
  test("RNN shape and loss") {
    val output = RNN(
      weightXh = param(ATen.ones(Array(2, 4), TensorOptions.dtypeDouble)),
      weightHh = param(ATen.ones(Array(4, 4), TensorOptions.dtypeDouble)),
      weightHq = param(ATen.ones(Array(4, 3), TensorOptions.dtypeDouble)),
      biasQ = param(ATen.ones(Array(3), TensorOptions.dtypeDouble)),
      biasH = param(ATen.ones(Array(4), TensorOptions.dtypeDouble)),
      dropout = 0d,
      train = true
    ).forward1(const(NDArray.tensorFromNDArray(nd2x3x2)), None)._1.value
    assert(output.shape == List(2, 3, 3))
    val target = TensorHelpers.fromLongMat(mat.ones(2, 3).map(_.toLong))
    val loss = LossFunctions
      .SequenceNLL(
        3,
        ATen.ones(Array(3), TensorOptions.dtypeDouble())
      )(const(output), target)
      ._1
      .value
    assert(TensorHelpers.toMat(loss).raw(0) == -4.9760101917362025)
  }

  test1("gradient clipping") { cuda =>
    val topt =
      if (cuda) TensorOptions.dtypeDouble().cuda
      else TensorOptions.dtypeDouble()
    val t = ATen.ones(Array(1, 2, 3), topt)
    val t2 = ATen.ones(Array(1, 2), topt)
    gradientClippingInPlace(Seq(Some(t), Some(t2)), 2.1)
    assert(
      NDArray.tensorToNDArray(t).toVec.roundTo(4) == NDArray(
        Array(0.7424621202458749, 0.7424621202458749, 0.7424621202458749,
          0.7424621202458749, 0.7424621202458749, 0.7424621202458749),
        List(1, 2, 3)
      ).toVec.roundTo(4)
    )
  }

  test("load params") {

    val mod = Sequential(
      Linear(in = 30, out = 3, TensorOptions.dtypeDouble()),
      Linear(in = 3, out = 2, TensorOptions.dtypeDouble()),
      Fun(_.logSoftMax(dim = 1))
    )

    val parameters = mod.state.map(_._1.value)

    val loaded = mod.load(parameters)
    assert(loaded.state.map(_._1.value) == parameters)
    val p2 = parameters.map { t => ATen.mul_1(t, 3d) }
    val loaded2 = mod.load(p2)

    assert(loaded2.state.map(_._1.value) == p2)

  }

  test("cyclic scheduler") {
    val sch = LearningRateSchedule.cyclicSchedule(10d, 18L)
    val rates = 0 to 40 map (i => sch(i.toLong))
    assert(
      rates == Vector(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 10.0,
        9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
        8.0, 9.0, 10.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 1.0, 2.0, 3.0,
        4.0, 5.0)
    )
  }

}

case class LogisticRegression1(dim: Int, k: Int, y: Variable) extends Module {

  val mat2x3_2 = Mat(Vec(-1d, 2d), Vec(3d, -4d), Vec(5d, 6d))
  val w = param(TensorHelpers.fromMat(mat2x3_2))

  def forward(x: Variable): Variable =
    ((x.mm(w)).logSoftMax(dim = 1).crossEntropy(y).sum + w.squaredFrobenius)

  override def load(parameters: Seq[Tensor]) = this

  override def state: Seq[(Variable, PTag)] =
    Nil

}
case class LogisticRegression2(dim: Int, k: Int, y: Variable) extends Module {

  val mod = Sequential(
    Linear(
      param(ATen.ones(Array(k, dim), y.options)),
      Some(param(ATen.ones(Array(1, k), y.options)))
    ),
    Fun(_.logSoftMax(dim = 1))
  )

  def forward(x: Variable): Variable =
    mod.forward(x).crossEntropy(y).sum +
      mod.parameters
        .map(_._1.squaredFrobenius)
        .reduce(_ + _)

  override def state: Seq[(Variable, PTag)] =
    mod.state

  override def load(parameters: Seq[Tensor]) = mod.load(parameters)

}

case class Mlp1(dim: Int, k: Int, y: Variable) extends Module {

  override def load(parameters: Seq[Tensor]) = mod.load(parameters)

  val mod = Sequential(
    Linear(
      param(ATen.ones(Array(32, dim), y.options)),
      Some(param(ATen.ones(Array(1, 32), y.options)))
    ),
    Fun(_.logSoftMax(dim = 1)),
    Fun(_.gelu),
    Linear(
      param(ATen.ones(Array(k, 32), y.options)),
      Some(param(ATen.ones(Array(1, k), y.options)))
    )
  )

  def forward(x: Variable): Variable =
    mod.forward(x).crossEntropy(y).sum +
      mod.parameters
        .map(_._1.squaredFrobenius)
        .reduce(_ + _)

  override def state: Seq[(Variable, PTag)] =
    mod.state

}
