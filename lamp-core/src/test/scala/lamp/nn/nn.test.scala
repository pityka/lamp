package lamp.nn

import org.saddle._
import org.saddle.ops.BinOps._
import lamp.saddle._
import org.scalatest.funsuite.AnyFunSuite
import lamp.autograd.NDArraySyntax._
import aten.ATen
import lamp.autograd.{BatchNorm => _, BatchNorm2D => _, Embedding => _, _}
import org.scalatest.Tag
import lamp.Scope
import lamp.Sc
import lamp.util.NDArray
import lamp.STen
import lamp.STenOptions
import org.scalatest.compatible.Assertion

object CudaTest extends Tag("cuda")
object SlowTest extends Tag("slow")

final class NNSuite extends AnyFunSuite {
  aten.Tensor.manual_seed(13223L)
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
    test(
      id + ": gradient is correct",
      (if (cuda) List(CudaTest) else Nil): _*
    ) {

      Scope.root { implicit scope =>
        val d = const(lamp.saddle.fromMat(m, cuda))
        val module = moduleF(scope)

        {
          val module1 = moduleF(scope)
          val state = module1.state
          val modifiedState = state.map { case (v, _) =>
            v.value * (-1)
          }
          module1.load(modifiedState)
          (module1.state zip modifiedState).foreach { case ((st1, _), (st2)) =>
            assert(
              NDArray.tensorToNDArray(st1.value.value).toVec == NDArray
                .tensorToNDArray(st2.value)
                .toVec
            )
          }
        }

        val output = module.forward(d)
        val sum = output.sum
        val value = sum.value.toMat.raw(0)
        val gradAuto = module.gradients(sum).map(_.get).map(_.toMat)
        val gradNum = module.parameters.map { case (paramT, _) =>
          val oldparam = paramT.value.cloneTensor
          val param = paramT.value.toMat
          def f(p: Mat[Double]) = {
            val p2 = lamp.saddle.fromMat(p, cuda)
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
          val r =
            mat.zeros(param.numRows, param.numCols).mapRows { case (row, i) =>
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
        gradAuto.zip(gradNum).foreach { case (a, b) =>
          assert(a.roundTo(4) == b.roundTo(4))
        }
        assert(math.abs(value - expectedValue) < 1e-6)
        ()
      }
    }
  def testGradientAndValueND[T, M <: StatefulModule[
    Variable,
    Variable,
    T
  ]: Load](
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
    test(
      id + ": gradient is correct",
      (if (cuda) List(CudaTest) else Nil): _*
    ) {
      Scope.root { implicit scope =>
        val d = const(STen.owned(NDArray.tensorFromNDArray(m, cuda)))
        val module = moduleF(scope)

        {
          val module1 = moduleF(scope)
          val state = module1.state
          val modifiedState = state.map { case (v, _) =>
            v.value * (-1)
          }
          module1.load(modifiedState)
          (module1.state zip modifiedState).foreach { case ((st1, _), (st2)) =>
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
        val gradNum = module.parameters.map { case (paramT, _) =>
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
        gradAuto.zip(gradNum).foreach { case (a, b) =>
          assert(a.toVec.roundTo(4) == b.toVec.roundTo(4))
        }
        assert(Vec(value).roundTo(4) == Vec(expectedValue).roundTo(4))
        ()
      }
    }
  def testGradientAndValueNDGeneric[A, B, M <: GenericModule[A, B]: Load](
      id: String,
      cuda: Boolean = false
  )(
      m: NDArray[Double],
      inputToA: Constant => A,
      bToVar: B => Variable,
      moduleF: Scope => M with GenericModule[A, B],
      expectedValue: Double
  ) =
    test(
      id + ": gradient is correct",
      (if (cuda) List(CudaTest) else Nil): _*
    ) {

      Scope.root { implicit scope =>
        val d = const(STen.owned(NDArray.tensorFromNDArray(m, cuda)))
        val module = moduleF(scope)

        {
          val module1 = moduleF(scope)
          val state = module1.state
          val modifiedState = state.map { case (v, _) =>
            v.value * (-1)
          }
          module1.load(modifiedState)
          (module1.state zip modifiedState).foreach { case ((st1, _), (st2)) =>
            assert(
              NDArray.tensorToNDArray(st1.value.value).toVec == NDArray
                .tensorToNDArray(st2.value)
                .toVec
            )
          }
        }

        val output = bToVar(module.forward(inputToA(d)))
        val sum = output.sum
        val value = NDArray.tensorToNDArray(sum.value.value).data(0)
        val gradAuto =
          module.gradients(sum).map(_.get.value).map(NDArray.tensorToNDArray)
        val gradNum = module.parameters.map { case (paramT, _) =>
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
            bToVar(module.forward(inputToA(d))).sum.value.toMat.raw(0)
          }
          val eps = 1e-3
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
        gradAuto.zip(gradNum).foreach { case (a, b) =>
          assert(
            a.toVec.roundTo(6) == b.toVec.roundTo(6),
            s"${a.toVec.toSeq} ${b.toVec.toSeq}"
          )
        }
        assert(Vec(value).roundTo(4) == Vec(expectedValue).roundTo(4))
        ()
      }
    }
  def testGradientAndValueNDLong[T, M <: StatefulModule[
    Variable,
    Variable,
    T
  ]: Load](
      id: String,
      st: T,
      cuda: Boolean = false,
      gradientCheck: Boolean = true
  )(
      m: NDArray[Long],
      moduleF: Scope => M with StatefulModule[
        Variable,
        Variable,
        T
      ],
      expectedValue: Double
  ) =
    test(
      id + ": gradient is correct",
      (if (cuda) List(CudaTest) else Nil): _*
    ) {
      Scope.root { implicit scope =>
        val d = const(STen.owned(NDArray.tensorFromLongNDArray(m, cuda)))
        val module = moduleF(scope)

        {
          val module1 = moduleF(scope)
          val state = module1.state
          val modifiedState = state.map { case (v, _) =>
            v.value * -1d
          }
          module1.load(modifiedState)
          (module1.state zip modifiedState).foreach { case ((st1, _), (st2)) =>
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
        val gradNum = module.parameters.map { case (paramT, _) =>
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
            SaddleTensorHelpers
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
        if (gradientCheck) {
          gradAuto.zip(gradNum).foreach { case (a, b) =>
            assert(a.toVec.roundTo(4) == b.toVec.roundTo(4))
          }
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
      val linear = Linear(3, 1, STenOptions.d)
      val output = linear.forward(const(lamp.saddle.fromMat(mat2x3)))
      val sum = output.sum
      assert(output.value.sizes.toList == List(2, 1))
      val grad = linear.gradients(sum)
      assert(grad.size == 2)
      ()
    }
  }
  test("linear 3D") {
    Scope.root { implicit scope =>
      val linear = Linear(5, 2, STenOptions.d)
      val output =
        linear.forward(const(STen.ones(List(3, 4, 5), STenOptions.d)))
      val sum = output.sum
      assert(output.value.sizes.toList == List(3, 4, 2))
      val grad = linear.gradients(sum)
      assert(grad.size == 2)
      ()
    }
  }
  testGradientAndValue("Linear 0")(
    mat2x3,
    implicit pool =>
      Linear(
        param(STen.ones(List(3, 1), STenOptions.d)),
        Some(param(STen.ones(List(1), STenOptions.d)))
      ),
    23d
  )

  testGradientAndValue("WeightNormLinear 0")(
    mat2x3,
    implicit pool =>
      WeightNormLinear(
        param(STen.ones(List(1, 3), STenOptions.d)),
        param(STen.ones(List(1, 3), STenOptions.d)),
        Some(param(STen.ones(List(1), STenOptions.d)))
      ),
    23d
  )
  testGradientAndValue("Logistic 1")(
    mat3x2,
    implicit pool =>
      LogisticRegression1(2, 3, const(lamp.saddle.fromMat(mat.ident(3))))(
        pool
      ),
    151.0000008318073
  )
  testGradientAndValue("Logistic 2")(
    mat3x2,
    implicit pool =>
      LogisticRegression2(2, 3, const(lamp.saddle.fromMat(mat.ident(3))))(
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
        const(lamp.saddle.fromMat(mat.ident(3), cuda = true))
      )(pool),
    12.295836866004326
  )
  testGradientAndValue("Mlp1 ", false)(
    mat3x2,
    implicit pool =>
      Mlp1(
        2,
        3,
        const(lamp.saddle.fromMat(mat.ident(3), cuda = false))
      )(pool),
    192.08796576929555
  )
  testGradientAndValue("Mlp1 - cuda", true)(
    mat3x2,
    implicit pool =>
      Mlp1(
        2,
        3,
        const(lamp.saddle.fromMat(mat.ident(3), cuda = true))
      )(pool),
    192.08796576929555
  )

  testGradientAndValueND("Conv1D ", (), false)(
    nd1x2x3,
    implicit pool =>
      Conv1D(
        param(STen.rand(List(1, 2, 3), STenOptions.d)),
        param(STen.rand(List(1), STenOptions.d)),
        stride = 1,
        padding = 0,
        dilation = 1,
        groups = 1
      ).lift,
    10.7941
  )
  testGradientAndValueND("Conv1D/cuda ", (), true)(
    nd1x2x3,
    implicit pool =>
      Conv1D(
        param(STen.rand(List(1, 2, 3), STenOptions.d.cudaIndex(0))),
        param(STen.rand(List(1), STenOptions.d.cuda)),
        stride = 1,
        padding = 0,
        dilation = 1,
        groups = 1
      ).lift,
    79.2702
  )
  testGradientAndValueND("Conv2D ", (), false)(
    nd1x2x3x3,
    implicit pool =>
      Conv2D(
        param(STen.rand(List(1, 2, 3, 3), STenOptions.d)),
        param(STen.rand(List(1), STenOptions.d)),
        stride = 1,
        padding = 0,
        dilation = 1,
        groups = 1
      ).lift,
    100.9049
  )
  testGradientAndValueND("Conv2D/cuda ", (), true)(
    nd1x2x3x3,
    implicit pool =>
      Conv2D(
        param(STen.rand(List(1, 2, 3, 3), STenOptions.d.cuda)),
        param(STen.rand(List(1), STenOptions.d.cuda)),
        stride = 1,
        padding = 0,
        dilation = 1,
        groups = 1
      ).lift,
    73.9732
  )
  testGradientAndValueND("BatchNorm ", (), false)(
    nd1x2x3x3,
    implicit pool =>
      BatchNorm(
        18,
        STenOptions.d
      ).lift,
    0d
  )
  testGradientAndValueND("BatchNorm 2D", (), false)(
    nd1x2x3x3,
    implicit pool =>
      BatchNorm2D(
        2,
        STenOptions.d
      ).lift,
    0d
  )

  testGradientAndValueNDLong("Embedding ", (), false)(
    nd2x3L,
    implicit pool =>
      Embedding(
        weights = param(STen.rand(List(2, 4), STenOptions.d))
      ).lift,
    9.1274
  )
  testGradientAndValueND("RNN ", Option.empty[Variable], false)(
    nd2x3x2,
    implicit pool =>
      RNN(
        weightXh = param(STen.rand(List(2, 4), STenOptions.d)),
        weightHh = param(STen.rand(List(4, 4), STenOptions.d)),
        biasH = param(STen.rand(List(4), STenOptions.d))
      ),
    23.4551
  )
  testGradientAndValueNDLong(
    "FreeRunning ",
    ((), Option.empty[Variable]),
    false,
    false
  )(
    nd2x3L,
    implicit pool => {
      val rnn = statefulSequence(
        Embedding
          .apply(weights = param(STen.rand(List(7, 4), STenOptions.d)))
          .lift,
        RNN(
          weightXh = param(STen.rand(List(4, 4), STenOptions.d)),
          weightHh = param(STen.rand(List(4, 4), STenOptions.d)),
          biasH = param(STen.rand(List(4), STenOptions.d))
        )
      )
      FreeRunningRNN(rnn, 3)
    },
    35.6161
  )
  testGradientAndValueND("SeqLinear ", (), false)(
    nd2x3x2,
    implicit pool =>
      SeqLinear(
        weight = param(STen.rand(List(2, 4), STenOptions.d)),
        bias = param(STen.rand(List(4), STenOptions.d))
      ).lift,
    123.3801
  )
  testGradientAndValueND("GRU ", Option.empty[Variable], false)(
    nd2x3x2,
    implicit pool =>
      GRU(
        weightXh = param(STen.rand(List(2, 4), STenOptions.d)),
        weightXr = param(STen.rand(List(2, 4), STenOptions.d)),
        weightXz = param(STen.rand(List(2, 4), STenOptions.d)),
        weightHh = param(STen.rand(List(4, 4), STenOptions.d)),
        weightHz = param(STen.rand(List(4, 4), STenOptions.d)),
        weightHr = param(STen.rand(List(4, 4), STenOptions.d)),
        biasH = param(STen.rand(List(4), STenOptions.d)),
        biasZ = param(STen.rand(List(4), STenOptions.d)),
        biasR = param(STen.rand(List(4), STenOptions.d))
      ),
    1.9875
  )
  testGradientAndValueND("LSTM ", Option.empty[(Variable, Variable)], false)(
    nd2x3x2,
    implicit pool =>
      LSTM(
        weightXi = param(STen.rand(List(2, 4), STenOptions.d)),
        weightXo = param(STen.rand(List(2, 4), STenOptions.d)),
        weightXf = param(STen.rand(List(2, 4), STenOptions.d)),
        weightXc = param(STen.rand(List(2, 4), STenOptions.d)),
        weightHi = param(STen.rand(List(4, 4), STenOptions.d)),
        weightHo = param(STen.rand(List(4, 4), STenOptions.d)),
        weightHf = param(STen.rand(List(4, 4), STenOptions.d)),
        weightHc = param(STen.rand(List(4, 4), STenOptions.d)),
        biasI = param(STen.rand(List(4), STenOptions.d)),
        biasO = param(STen.rand(List(4), STenOptions.d)),
        biasF = param(STen.rand(List(4), STenOptions.d)),
        biasC = param(STen.rand(List(4), STenOptions.d))
      ),
    17.7721
  )
  test("RNN shape and loss") {
    Scope.root { implicit scope =>
      val output = RNN(
        weightXh = param(STen.ones(List(2, 4), STenOptions.d)),
        weightHh = param(STen.ones(List(4, 4), STenOptions.d)),
        biasH = param(STen.ones(List(4), STenOptions.d))
      ).forward1(const(STen.owned(NDArray.tensorFromNDArray(nd2x3x2))), None)
        ._1
        .value
      assert(output.shape == List(2, 3, 4))
      val target =
        STen.owned(
          SaddleTensorHelpers.fromLongMat(mat.ones(2, 3).map(_.toLong))
        )
      val loss = LossFunctions
        .SequenceNLL(
          4,
          STen.ones(List(4), STenOptions.d)
        )(const(output), target)
        ._1
        .value
      assert(
        SaddleTensorHelpers.toMat(loss.value).raw(0) == -0.9940025479340507
      )
      ()
    }
  }

  test1("gradient clipping") { cuda =>
    implicit val AssertionIsMovable: lamp.EmptyMovable[Assertion] =
      lamp.Movable.empty[Assertion]
    Scope.root { implicit scope =>
      val topt =
        if (cuda) STenOptions.d.cuda
        else STenOptions.d
      val t = STen.ones(List(1, 2, 3), topt)
      val t2 = STen.ones(List(1, 2), topt)
      gradientClippingInPlace(
        Seq(Some(t), Some(t2)),
        STen.scalarDouble(2.1, topt)
      )
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
        Linear(in = 30, out = 3, STenOptions.d),
        Linear(in = 3, out = 2, STenOptions.d),
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
    val rates = 0 to 40 map (i => sch.learningRateFactor((), i.toLong, None))
    assert(
      rates.map(_._2) == Vector(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
        10.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0,
        6.0, 7.0, 8.0, 9.0, 10.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 1.0,
        2.0, 3.0, 4.0, 5.0)
    )
  }

  testGradientAndValueNDGeneric[
    (Variable, Option[STen]),
    Variable,
    TransformerEncoder
  ](
    "transformer encoder ",
    false
  )(
    nd2x3x2,
    v =>
      (
        v,
        Option(
          STen.owned(NDArray.tensorFromLongNDArray(nd2x3L, false))(Scope.free)
        )
      ),
    v => v,
    implicit pool => {
      aten.Tensor.manual_seed(123)

      val in = 2
      val attentionHiddenPerHeadDim = 4
      val attentionNumHeads = 5
      val mlpHiddenDim = 7
      val dropout = 0d
      // val padToken = -999L
      val tOpt = STenOptions.d
      TransformerEncoder.apply(
        List(
          TransformerEncoderBlock(
            attention = MultiheadAttention(
              wQ = param(
                STen.ones(
                  List(in, attentionHiddenPerHeadDim * attentionNumHeads),
                  tOpt
                ) * 2
              ),
              wK = param(
                STen.ones(
                  List(in, attentionHiddenPerHeadDim * attentionNumHeads),
                  tOpt
                ) * 2
              ),
              wV = param(
                STen.ones(
                  List(in, attentionHiddenPerHeadDim * attentionNumHeads),
                  tOpt
                ) * 2
              ),
              wO = param(
                STen.ones(
                  List(attentionHiddenPerHeadDim * attentionNumHeads, in),
                  tOpt
                ) * 2
              ),
              dropout = dropout,
              train = true,
              numHeads = attentionNumHeads,
              // padToken = padToken,
              linearized = false,
              causalMask = false
            ),
            gptOrder = false,
            layerNorm1 = lamp.nn.LayerNorm(normalizedShape = List(2L), tOpt),
            layerNorm2 = lamp.nn.LayerNorm(List(2L), tOpt),
            w1 = param(STen.ones(List(in, mlpHiddenDim), tOpt) * 2),
            b1 = param(STen.ones(List(1, mlpHiddenDim), tOpt) * 2),
            w2 = param(STen.ones(List(mlpHiddenDim, in), tOpt) * 2),
            b2 = param(STen.ones(List(1, in), tOpt)),
            scale1 = param(STen.ones(List(in.toLong), tOpt)),
            scale2 = param(STen.ones(List(in.toLong), tOpt)),
            dropout = dropout,
            train = true
          )
        )
      )
    },
    0d
  )
  testGradientAndValueNDGeneric[
    (Variable, Option[STen]),
    Variable,
    TransformerEncoder
  ](
    "linearized transformer encoder ",
    false
  )(
    nd2x3x2,
    v =>
      (
        v,
        Option(
          STen.owned(NDArray.tensorFromLongNDArray(nd2x3L, false))(Scope.free)
        )
      ),
    v => v,
    implicit pool => {
      aten.Tensor.manual_seed(123)

      val in = 2
      val attentionHiddenPerHeadDim = 4
      val attentionNumHeads = 5
      val mlpHiddenDim = 7
      val dropout = 0d
      // val padToken = -999L
      val tOpt = STenOptions.d
      TransformerEncoder.apply(
        List(
          TransformerEncoderBlock(
            attention = MultiheadAttention(
              wQ = param(
                STen.ones(
                  List(in, attentionHiddenPerHeadDim * attentionNumHeads),
                  tOpt
                ) * 2
              ),
              wK = param(
                STen.ones(
                  List(in, attentionHiddenPerHeadDim * attentionNumHeads),
                  tOpt
                ) * 2
              ),
              wV = param(
                STen.ones(
                  List(in, attentionHiddenPerHeadDim * attentionNumHeads),
                  tOpt
                ) * 2
              ),
              wO = param(
                STen.ones(
                  List(attentionHiddenPerHeadDim * attentionNumHeads, in),
                  tOpt
                ) * 2
              ),
              dropout = dropout,
              train = true,
              numHeads = attentionNumHeads,
              linearized = true,
              causalMask = false
            ),
            gptOrder = false,
            layerNorm1 = LayerNorm(List(2), tOpt),
            layerNorm2 = LayerNorm(List(2), tOpt),
            w1 = param(STen.ones(List(in, mlpHiddenDim), tOpt) * 2),
            b1 = param(STen.ones(List(1, mlpHiddenDim), tOpt) * 2),
            w2 = param(STen.ones(List(mlpHiddenDim, in), tOpt) * 2),
            b2 = param(STen.ones(List(1, in), tOpt)),
            scale1 = param(STen.ones(List(in.toLong), tOpt)),
            scale2 = param(STen.ones(List(in.toLong), tOpt)),
            dropout = dropout,
            train = true
          )
        )
      )
    },
    0d
  )

}

case class LogisticRegression1(dim: Int, k: Int, y: Variable)(
    pool: Scope
) extends Module {
  val mat2x3_2 = Mat(Vec(-1d, 2d), Vec(3d, -4d), Vec(5d, 6d))
  val w = param(lamp.saddle.fromMat(mat2x3_2)(pool))(pool)

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
      param(STen.ones(List(dim, k), y.options(pool))(pool))(pool),
      Some(param(STen.ones(List(1, k), y.options(pool))(pool))(pool))
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
      param(STen.ones(List(dim, 32), y.options(pool))(pool))(pool),
      Some(param(STen.ones(List(1, 32), y.options(pool))(pool))(pool))
    ),
    Fun(scope => input => input.logSoftMax(dim = 1)(scope)),
    Fun(scope => input => input.gelu(scope)),
    Linear(
      param(STen.ones(List(32, k), y.options(pool))(pool))(pool),
      Some(param(STen.ones(List(1, k), y.options(pool))(pool))(pool))
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
