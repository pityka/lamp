package lamp.nn

import org.saddle._
import org.saddle.ops.BinOps._
import org.scalatest.funsuite.AnyFunSuite
import aten.ATen
import lamp.autograd._
import aten.TensorOptions
import org.scalatest.Tag
import lamp.TensorHelpers
import lamp.Scope
import lamp.Sc
import lamp.util.NDArray
import lamp.STen

object CudaTest extends Tag("cuda")
object SlowTest extends Tag("slow")

class NNSuite extends AnyFunSuite {

  def test1(id: String)(fun: Boolean => Unit) = {
    test(id) { fun(false) }
    test(id + "/CUDA", CudaTest) { fun(true) }
  }
  def testGradientAndValue[M <: Module: Load](
      id: String,
      cuda: Boolean = false
  )(
      m: Mat[Double],
      moduleF: Scope => M with Module,
      expectedValue: Double
  ) =
    test(id + ": gradient is correct", (if (cuda) List(CudaTest) else Nil): _*) {

      Scope.root { implicit scope =>
        val d = const(STen.fromMat(m, cuda))
        val module = moduleF(scope)

        {
          val module1 = moduleF(scope)
          val state = module1.state
          val modifiedState = state.map {
            case (v, _) =>
              v.value * (-1)
          }
          module1.load(modifiedState)
          (module1.state zip modifiedState).foreach {
            case ((st1, _), (st2)) =>
              assert(
                NDArray.tensorToNDArray(st1.value.value).toVec == NDArray
                  .tensorToNDArray(st2.value)
                  .toVec
              )
          }
        }

        val output = module.forward(d)
        val sum = output.sum
        val value = sum.toMat.raw(0)
        val gradAuto = module.gradients(sum).map(_.get).map(_.toMat)
        val gradNum = module.parameters.map {
          case (paramT, _) =>
            val oldparam = paramT.value.cloneTensor
            val param = paramT.toMat
            def f(p: Mat[Double]) = {
              val p2 = STen.fromMat(p, cuda)
              paramT.value.zero_()
              ATen.add_out(
                paramT.value.value,
                paramT.value.value,
                ATen._unsafe_view(p2.value, paramT.sizes.toArray),
                1d
              )
              module.forward(d).sum.value.toMat.raw(0)
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
            paramT.value.zero_()
            ATen.add_out(
              paramT.value.value,
              paramT.value.value,
              oldparam.value,
              1d
            )
            r
        }
        assert(gradAuto.size == gradNum.size)
        gradAuto.zip(gradNum).foreach {
          case (a, b) =>
            assert(a.roundTo(4) == b.roundTo(4))
        }
        assert(math.abs(value - expectedValue) < 1e-6)
        ()
      }
    }
  def testGradientAndValueND[T, M <: StatefulModule[Variable, Variable, T]: Load](
      id: String,
      st: T,
      cuda: Boolean = false
  )(
      m: NDArray[Double],
      moduleF: Scope => M with StatefulModule[
        Variable,
        Variable,
        T
      ],
      expectedValue: Double
  ) =
    test(id + ": gradient is correct", (if (cuda) List(CudaTest) else Nil): _*) {
      Scope.root { implicit scope =>
        val d = const(STen.owned(NDArray.tensorFromNDArray(m, cuda)))
        val module = moduleF(scope)

        {
          val module1 = moduleF(scope)
          val state = module1.state
          val modifiedState = state.map {
            case (v, _) =>
              v.value * (-1)
          }
          module1.load(modifiedState)
          (module1.state zip modifiedState).foreach {
            case ((st1, _), (st2)) =>
              assert(
                NDArray.tensorToNDArray(st1.value.value).toVec == NDArray
                  .tensorToNDArray(st2.value)
                  .toVec
              )
          }
        }

        val output = module.forward((d, st))._1
        val sum = output.sum
        val value = NDArray.tensorToNDArray(sum.value.value).data(0)
        val gradAuto =
          module.gradients(sum).map(_.get.value).map(NDArray.tensorToNDArray)
        val gradNum = module.parameters.map {
          case (paramT, _) =>
            val oldparam = paramT.value.cloneTensor
            val param = NDArray.tensorToNDArray(paramT.value.value)
            def f(p: NDArray[Double]) = {
              val p2 = NDArray.tensorFromNDArray(p, cuda)
              paramT.value.zero_()
              ATen.add_out(
                paramT.value.value,
                paramT.value.value,
                p2,
                1d
              )
              module.forward((d, st))._1.sum.value.toMat.raw(0)
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
            paramT.value.zero_()
            ATen.add_out(
              paramT.value.value,
              paramT.value.value,
              oldparam.value,
              1d
            )
            r
        }
        assert(gradAuto.size == gradNum.size)
        gradAuto.zip(gradNum).foreach {
          case (a, b) =>
            assert(a.toVec.roundTo(4) == b.toVec.roundTo(4))
        }
        assert(Vec(value).roundTo(4) == Vec(expectedValue).roundTo(4))
        ()
      }
    }
  def testGradientAndValueNDLong[T, M <: StatefulModule[Variable, Variable, T]: Load](
      id: String,
      st: T,
      cuda: Boolean = false
  )(
      m: NDArray[Long],
      moduleF: Scope => M with StatefulModule[
        Variable,
        Variable,
        T
      ],
      expectedValue: Double
  ) =
    test(id + ": gradient is correct", (if (cuda) List(CudaTest) else Nil): _*) {
      Scope.root { implicit scope =>
        val d = const(STen.owned(NDArray.tensorFromLongNDArray(m, cuda)))
        val module = moduleF(scope)

        {
          val module1 = moduleF(scope)
          val state = module1.state
          val modifiedState = state.map {
            case (v, _) =>
              v.value * -1d
          }
          module1.load(modifiedState)
          (module1.state zip modifiedState).foreach {
            case ((st1, _), (st2)) =>
              assert(
                NDArray.tensorToNDArray(st1.value.value).toVec == NDArray
                  .tensorToNDArray(st2.value)
                  .toVec
              )
          }
        }

        val output = module.forward((d, st))._1
        val sum = output.sum
        val value = NDArray.tensorToNDArray(sum.value.value).data(0)
        val gradAuto =
          module.gradients(sum).map(_.get.value).map(NDArray.tensorToNDArray)
        val gradNum = module.parameters.map {
          case (paramT, _) =>
            val oldparam = ATen.clone(paramT.value.value)
            val param = NDArray.tensorToNDArray(paramT.value.value)
            def f(p: NDArray[Double]) = {
              val p2 = NDArray.tensorFromNDArray(p, cuda)
              ATen.zero_(paramT.value.value)
              ATen.add_out(
                paramT.value.value,
                paramT.value.value,
                p2,
                1d
              )
              TensorHelpers
                .toMat(module.forward((d, st))._1.sum.value.value)
                .raw(0)
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
            ATen.zero_(paramT.value.value)
            ATen.add_out(paramT.value.value, paramT.value.value, oldparam, 1d)
            r
        }
        assert(gradAuto.size == gradNum.size)
        gradAuto.zip(gradNum).foreach {
          case (a, b) =>
            assert(a.toVec.roundTo(4) == b.toVec.roundTo(4))
        }
        assert(Vec(value).roundTo(4) == Vec(expectedValue).roundTo(4))
        ()
      }
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
    Scope.root { implicit scope =>
      val linear = Linear(3, 1, TensorOptions.dtypeDouble())
      val output = linear.forward(const(STen.fromMat(mat2x3)))
      val sum = output.sum
      assert(output.value.sizes.toList == List(2, 1))
      val grad = linear.gradients(sum)
      assert(grad.size == 2)
      ()
    }
  }
  testGradientAndValue("Linear 0")(
    mat2x3,
    implicit pool =>
      Linear(
        param(STen.ones(Array(1, 3), TensorOptions.dtypeDouble)),
        Some(param(STen.ones(Array(1), TensorOptions.dtypeDouble)))
      ),
    23d
  )

  testGradientAndValue("WeightNormLinear 0")(
    mat2x3,
    implicit pool =>
      WeightNormLinear(
        param(STen.ones(Array(1, 3), TensorOptions.dtypeDouble)),
        param(STen.ones(Array(1, 3), TensorOptions.dtypeDouble)),
        Some(param(STen.ones(Array(1), TensorOptions.dtypeDouble)))
      ),
    23d
  )
  testGradientAndValue("Logistic 1")(
    mat3x2,
    implicit pool =>
      LogisticRegression1(2, 3, const(STen.fromMat(mat.ident(3))))(
        pool
      ),
    151.0000008318073
  )
  testGradientAndValue("Logistic 2")(
    mat3x2,
    implicit pool =>
      LogisticRegression2(2, 3, const(STen.fromMat(mat.ident(3))))(
        pool
      ),
    12.295836866004327
  )
  testGradientAndValue("Logistic 2 - cuda", true)(
    mat3x2,
    implicit pool =>
      LogisticRegression2(
        2,
        3,
        const(STen.fromMat(mat.ident(3), cuda = true))
      )(pool),
    12.295836866004326
  )
  testGradientAndValue("Mlp1 ", false)(
    mat3x2,
    implicit pool =>
      Mlp1(
        2,
        3,
        const(STen.fromMat(mat.ident(3), cuda = false))
      )(pool),
    192.08796576929555
  )
  testGradientAndValue("Mlp1 - cuda", true)(
    mat3x2,
    implicit pool =>
      Mlp1(
        2,
        3,
        const(STen.fromMat(mat.ident(3), cuda = true))
      )(pool),
    192.08796576929555
  )

  testGradientAndValueND("Conv1D ", (), false)(
    nd1x2x3,
    implicit pool =>
      Conv1D(
        param(STen.ones(Array(1, 2, 3), TensorOptions.dtypeDouble)),
        param(STen.ones(Array(1), TensorOptions.dtypeDouble)),
        stride = 1,
        padding = 0,
        dilation = 1,
        groups = 1
      ).lift,
    22d
  )
  testGradientAndValueND("Conv1D/cuda ", (), true)(
    nd1x2x3,
    implicit pool =>
      Conv1D(
        param(STen.ones(Array(1, 2, 3), TensorOptions.dtypeDouble.cuda)),
        param(STen.ones(Array(1), TensorOptions.dtypeDouble.cuda)),
        stride = 1,
        padding = 0,
        dilation = 1,
        groups = 1
      ).lift,
    22d
  )
  testGradientAndValueND("Conv2D ", (), false)(
    nd1x2x3x3,
    implicit pool =>
      Conv2D(
        param(STen.ones(Array(1, 2, 3, 3), TensorOptions.dtypeDouble)),
        param(STen.ones(Array(1), TensorOptions.dtypeDouble)),
        stride = 1,
        padding = 0,
        dilation = 1,
        groups = 1
      ).lift,
    154d
  )
  testGradientAndValueND("Conv2D/cuda ", (), true)(
    nd1x2x3x3,
    implicit pool =>
      Conv2D(
        param(STen.ones(Array(1, 2, 3, 3), TensorOptions.dtypeDouble.cuda)),
        param(STen.ones(Array(1), TensorOptions.dtypeDouble.cuda)),
        stride = 1,
        padding = 0,
        dilation = 1,
        groups = 1
      ).lift,
    154d
  )
  testGradientAndValueND("BatchNorm ", (), false)(
    nd1x2x3x3,
    implicit pool =>
      BatchNorm(
        18,
        TensorOptions.dtypeDouble()
      ).lift,
    0d
  )
  testGradientAndValueND("BatchNorm 2D", (), false)(
    nd1x2x3x3,
    implicit pool =>
      BatchNorm2D(
        2,
        TensorOptions.dtypeDouble()
      ).lift,
    0d
  )

  testGradientAndValueNDLong("Embedding ", (), false)(
    nd2x3L,
    implicit pool =>
      Embedding(
        weights = param(STen.ones(Array(2, 4), TensorOptions.dtypeDouble))
      ).lift,
    24d
  )
  testGradientAndValueND("RNN ", Option.empty[Variable], false)(
    nd2x3x2,
    implicit pool =>
      RNN(
        weightXh = param(STen.ones(Array(2, 4), TensorOptions.dtypeDouble)),
        weightHh = param(STen.ones(Array(4, 4), TensorOptions.dtypeDouble)),
        biasH = param(STen.ones(Array(4), TensorOptions.dtypeDouble))
      ),
    23.8561
  )
  testGradientAndValueNDLong(
    "FreeRunning ",
    ((), Option.empty[Variable]),
    false
  )(
    nd2x3L,
    implicit pool => {
      val rnn = statefulSequence(
        Embedding
          .apply(weights =
            param(STen.ones(Array(7, 4), TensorOptions.dtypeDouble))
          )
          .lift,
        RNN(
          weightXh = param(STen.ones(Array(4, 4), TensorOptions.dtypeDouble)),
          weightHh = param(STen.ones(Array(4, 4), TensorOptions.dtypeDouble)),
          biasH = param(STen.ones(Array(4), TensorOptions.dtypeDouble))
        )
      )
      FreeRunningRNN(rnn, 3)
    },
    36d
  )
  testGradientAndValueND("SeqLinear ", (), false)(
    nd2x3x2,
    implicit pool =>
      SeqLinear(
        weight = param(STen.ones(Array(2, 4), TensorOptions.dtypeDouble)),
        bias = param(STen.ones(Array(4), TensorOptions.dtypeDouble))
      ).lift,
    288d
  )
  testGradientAndValueND("GRU ", Option.empty[Variable], false)(
    nd2x3x2,
    implicit pool =>
      GRU(
        weightXh = param(STen.ones(Array(2, 4), TensorOptions.dtypeDouble)),
        weightXr = param(STen.ones(Array(2, 4), TensorOptions.dtypeDouble)),
        weightXz = param(STen.ones(Array(2, 4), TensorOptions.dtypeDouble)),
        weightHh = param(STen.ones(Array(4, 4), TensorOptions.dtypeDouble)),
        weightHz = param(STen.ones(Array(4, 4), TensorOptions.dtypeDouble)),
        weightHr = param(STen.ones(Array(4, 4), TensorOptions.dtypeDouble)),
        biasH = param(STen.ones(Array(4), TensorOptions.dtypeDouble)),
        biasZ = param(STen.ones(Array(4), TensorOptions.dtypeDouble)),
        biasR = param(STen.ones(Array(4), TensorOptions.dtypeDouble))
      ),
    0.9395
  )
  testGradientAndValueND("LSTM ", Option.empty[(Variable, Variable)], false)(
    nd2x3x2,
    implicit pool =>
      LSTM(
        weightXi = param(STen.ones(Array(2, 4), TensorOptions.dtypeDouble)),
        weightXo = param(STen.ones(Array(2, 4), TensorOptions.dtypeDouble)),
        weightXf = param(STen.ones(Array(2, 4), TensorOptions.dtypeDouble)),
        weightXc = param(STen.ones(Array(2, 4), TensorOptions.dtypeDouble)),
        weightHi = param(STen.ones(Array(4, 4), TensorOptions.dtypeDouble)),
        weightHo = param(STen.ones(Array(4, 4), TensorOptions.dtypeDouble)),
        weightHf = param(STen.ones(Array(4, 4), TensorOptions.dtypeDouble)),
        weightHc = param(STen.ones(Array(4, 4), TensorOptions.dtypeDouble)),
        biasI = param(STen.ones(Array(4), TensorOptions.dtypeDouble)),
        biasO = param(STen.ones(Array(4), TensorOptions.dtypeDouble)),
        biasF = param(STen.ones(Array(4), TensorOptions.dtypeDouble)),
        biasC = param(STen.ones(Array(4), TensorOptions.dtypeDouble))
      ),
    20.0321
  )
  test("RNN shape and loss") {
    Scope.root { implicit scope =>
      val output = RNN(
        weightXh = param(STen.ones(Array(2, 4), TensorOptions.dtypeDouble)),
        weightHh = param(STen.ones(Array(4, 4), TensorOptions.dtypeDouble)),
        biasH = param(STen.ones(Array(4), TensorOptions.dtypeDouble))
      ).forward1(const(STen.owned(NDArray.tensorFromNDArray(nd2x3x2))), None)
        ._1
        .value
      assert(output.shape == List(2, 3, 4))
      val target =
        STen.owned(TensorHelpers.fromLongMat(mat.ones(2, 3).map(_.toLong)))
      val loss = LossFunctions
        .SequenceNLL(
          4,
          STen.ones(Array(4), TensorOptions.dtypeDouble())
        )(const(output), target)
        ._1
        .value
      assert(TensorHelpers.toMat(loss.value).raw(0) == -0.9940025479340507)
      ()
    }
  }

  test1("gradient clipping") { cuda =>
    Scope.root { implicit scope =>
      val topt =
        if (cuda) TensorOptions.dtypeDouble().cuda
        else TensorOptions.dtypeDouble()
      val t = STen.ones(Array(1, 2, 3), topt)
      val t2 = STen.ones(Array(1, 2), topt)
      gradientClippingInPlace(Seq(Some(t), Some(t2)), 2.1)
      assert(
        NDArray.tensorToNDArray(t.value).toVec.roundTo(4) == NDArray(
          Array(0.7424621202458749, 0.7424621202458749, 0.7424621202458749,
            0.7424621202458749, 0.7424621202458749, 0.7424621202458749),
          List(1, 2, 3)
        ).toVec.roundTo(4)
      )
    }
  }

  test("load params") {
    Scope.root { implicit scope =>
      val mod = sequence(
        Linear(in = 30, out = 3, TensorOptions.dtypeDouble()),
        Linear(in = 3, out = 2, TensorOptions.dtypeDouble()),
        Fun(scope => input => input.logSoftMax(dim = 1)(scope))
      )

      val parameters = mod.state.map(_._1.value)

      mod.load(parameters)
      assert(mod.state.map(_._1.value.toMat) == parameters.map(_.toMat))
      val p2 = parameters.map { t => t * 3 }
      mod.load(p2)

      assert(mod.state.map(_._1.value.toMat) == parameters.map(_.toMat))
      ()
    }
  }

  test("cyclic scheduler") {
    val sch = LearningRateSchedule.cyclicSchedule(10d, 18L)
    val rates = 0 to 40 map (i => sch.factor(i.toLong, None))
    assert(
      rates == Vector(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 10.0,
        9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
        8.0, 9.0, 10.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 1.0, 2.0, 3.0,
        4.0, 5.0)
    )
  }

}

case class LogisticRegression1(dim: Int, k: Int, y: Variable)(
    pool: Scope
) extends Module {
  val mat2x3_2 = Mat(Vec(-1d, 2d), Vec(3d, -4d), Vec(5d, 6d))
  val w = param(STen.fromMat(mat2x3_2)(pool))(pool)

  def forward[S: Sc](x: Variable): Variable =
    ((x.mm(w)).logSoftMax(dim = 1).crossEntropy(y).sum + w.squaredFrobenius)

  override def state =
    Nil

}
object LogisticRegression1 {
  implicit val load: Load[LogisticRegression1] =
    Load.identity[LogisticRegression1]
}
case class LogisticRegression2(dim: Int, k: Int, y: Variable)(
    pool: Scope
) extends Module {
  val mod = sequence(
    Linear(
      param(STen.ones(Array(k, dim), y.options)(pool))(pool),
      Some(param(STen.ones(Array(1, k), y.options)(pool))(pool))
    ),
    Fun(scope => input => input.logSoftMax(dim = 1)(scope))
  )

  def forward[S: Sc](x: Variable): Variable =
    mod.forward(x).crossEntropy(y).sum +
      mod.parameters
        .map(_._1.squaredFrobenius)
        .reduce(_ + _)

  override def state =
    Nil

}
object LogisticRegression2 {
  implicit val load: Load[LogisticRegression2] =
    Load.identity[LogisticRegression2]
}

case class Mlp1(dim: Int, k: Int, y: Variable)(
    pool: Scope
) extends Module {

  val mod = Sequential(
    Linear(
      param(STen.ones(Array(32, dim), y.options)(pool))(pool),
      Some(param(STen.ones(Array(1, 32), y.options)(pool))(pool))
    ),
    Fun(scope => input => input.logSoftMax(dim = 1)(scope)),
    Fun(scope => input => input.gelu(scope)),
    Linear(
      param(STen.ones(Array(k, 32), y.options)(pool))(pool),
      Some(param(STen.ones(Array(1, k), y.options)(pool))(pool))
    )
  )

  def forward[S: Sc](x: Variable): Variable =
    mod.forward(x).crossEntropy(y).sum +
      mod.parameters
        .map(_._1.squaredFrobenius)
        .reduce(_ + _)

  override def state =
    Nil

}
object Mlp1 {
  implicit val load: Load[Mlp1] = Load.identity[Mlp1]
}
