package lamp.nn

import org.saddle._
import org.saddle.ops.BinOps._
import lamp.saddle._
import org.scalatest.funsuite.AnyFunSuite
import lamp.autograd.NDArraySyntax._
import aten.ATen
import lamp.autograd.{ Embedding => _, _}
import org.scalatest.Tag
import lamp.Scope
import lamp.Sc
import lamp.util.NDArray
import lamp.STen
import lamp.STenOptions
import lamp.autograd.Autograd.BackpropForwardCache
import org.scalatest.compatible.Assertion
import lamp.CudaDevice
import lamp.CPU
import lamp.MPS

object CudaTest extends Tag("cuda")
object SlowTest extends Tag("slow")

final class NNSuite extends AnyFunSuite {
  aten.Tensor.manual_seed(13223L)

  test("NDArray") {
    val aDouble = NDArray(Array(1d,2d,3d,4d,5d,6d),List(2,3))
    val aLong = NDArray(Array(1d,2d,3d,4d,5d,6d).map(_.toLong),List(2,3))
    val cpuD = NDArray.tensorFromNDArray(aDouble)
    val cpuL = NDArray.tensorFromLongNDArray(aLong)
    val cpuF = ATen._cast_Float(cpuD,false)

    
    assert(NDArray.tensorToNDArray(cpuD).data.toList == aDouble.data.toList )
    assert(NDArray.tensorToNDArray(cpuF).data.toList == aDouble.data.toList )
    assert(NDArray.tensorToNDArray(cpuL).data.toList == aLong.data.toList )

    if (aten.Tensor.hasMps()) {
      val mpsD = lamp.MPS.to(ATen._cast_Float(cpuD,false)).cpu
      val mpsL = lamp.MPS.to(cpuL).cpu
      val mpsF = lamp.MPS.to(cpuF).cpu
      assert(NDArray.tensorToNDArray(mpsD).data.toList == aDouble.data.toList )
    assert(NDArray.tensorToNDArray(mpsF).data.toList == aDouble.data.toList )
    assert(NDArray.tensorToNDArray(mpsL).data.toList == aLong.data.toList )
    }
    
    
    
  }
    

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
        implicit val fw : BackpropForwardCache = BackpropForwardCache.simple
        val d = const(lamp.saddle.fromMat(m, cuda))
        val module = moduleF(scope)

        {
          val module1 = moduleF(scope)
          val state = module1.state
          val modifiedState = state.map { case (v, _) =>
            v.forward * (-1)
          }
          module1.load(modifiedState)
          (module1.state zip modifiedState).foreach { case ((st1, _), (st2)) =>
            assert(
              NDArray.tensorToNDArray(st1.forward.value).toVec == NDArray
                .tensorToNDArray(st2.value)
                .toVec
            )
          }
        }

        val output = module.forward(d)
        val sum = output.sum.persist
        val value = sum.forward.toMat.raw(0)
        val pd = ParameterGradients.empty
        val map2 = module.computeGradientsAndDestroyGraph(sum,pd.map.map(x => (x._1:Variable,x._2)),scope,true,fw,false)
        val gradAuto= module.parameters.map(v => map2(v._1)).map(_.toMat)
        val gradNum = module.parameters.map { case (paramT, _) =>
          val oldparam = paramT.forward.cloneTensor
          val param = paramT.forward.toMat
          def f(p: Mat[Double]) = {
            val p2 = lamp.saddle.fromMat(p, cuda)
            paramT.forward.zero_()
            ATen.add_out(
              paramT.forward.value,
              paramT.forward.value,
              ATen._unsafe_view(p2.value, paramT.eval.sizes.toArray),
              1d
            )
            module.forward(d).sum.forward.toMat.raw(0)
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
          paramT.forward.zero_()
          ATen.add_out(
            paramT.forward.value,
            paramT.forward.value,
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
 
      def testGradientAndValueND[ M <: Module : Load](
      id: String,
      cuda: Boolean = false
  )(
      m: NDArray[Double],
      moduleF: Scope => M ,
      expectedValue: Double
  ) =
    test(
      id + ": gradient is correct",
      (if (cuda) List(CudaTest) else Nil): _*
    ) {
      Scope.root { implicit scope =>
        implicit val fw : BackpropForwardCache = BackpropForwardCache.simple
        val d = const(STen.owned(NDArray.tensorFromNDArray(m, if (cuda) CudaDevice(0) else lamp.CPU)))
        val module = moduleF(scope)

        {
          val module1 = moduleF(scope)
          val state = module1.state
          val modifiedState = state.map { case (v, _) =>
            v.forward * (-1)
          }
          module1.load(modifiedState)
          (module1.state zip modifiedState).foreach { case ((st1, _), (st2)) =>
            assert(
              NDArray.tensorToNDArray(st1.forward.value).toVec == NDArray
                .tensorToNDArray(st2.value)
                .toVec
            )
          }
        }

        val output = module.forward((d))
        val sum = output.sum.persist
        val value = NDArray.tensorToNDArray(sum.forward.value).data(0)
        val gradAuto =
          module.computeGradientsAndDestroyGraph(sum,fw,false).map(_.value).map(NDArray.tensorToNDArray)
        val gradNum = module.parameters.map { case (paramT, _) =>
          val oldparam = paramT.forward.cloneTensor
          val param = NDArray.tensorToNDArray(paramT.forward.value)
          def f(p: NDArray[Double]) = {
            val p2 = NDArray.tensorFromNDArray(p, if (cuda) CudaDevice(0) else lamp.CPU)
            paramT.forward.zero_()
            ATen.add_out(
              paramT.forward.value,
              paramT.forward.value,
              p2,
              1d
            )
            module.forward((d)).sum.forward.toMat.raw(0)
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
          paramT.forward.zero_()
          ATen.add_out(
            paramT.forward.value,
            paramT.forward.value,
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
      def testGradientAndValueNDLong[ M <: GenericModule[STen,Variable] : Load](
      id: String,
      cuda: Boolean = false
  )(
      m: NDArray[Long],
      moduleF: Scope => M ,
      expectedValue: Double
  ) =
    test(
      id + ": gradient is correct",
      (if (cuda) List(CudaTest) else Nil): _*
    ) {
      Scope.root { implicit scope =>
        implicit val fw : BackpropForwardCache = BackpropForwardCache.simple
        val d = (STen.owned(NDArray.tensorFromLongNDArray(m, if (cuda) CudaDevice(0) else lamp.CPU)))
        val module = moduleF(scope)

        {
          val module1 = moduleF(scope)
          val state = module1.state
          val modifiedState = state.map { case (v, _) =>
            v.forward * (-1)
          }
          module1.load(modifiedState)
          (module1.state zip modifiedState).foreach { case ((st1, _), (st2)) =>
            assert(
              NDArray.tensorToNDArray(st1.forward.value).toVec == NDArray
                .tensorToNDArray(st2.value)
                .toVec
            )
          }
        }

        val output = module.forward((d))
        val sum = output.sum.persist
        val value = NDArray.tensorToNDArray(sum.forward.value).data(0)
        val gradAuto =
          module.computeGradientsAndDestroyGraph(sum,fw,false).map(_.value).map(NDArray.tensorToNDArray)
        val gradNum = module.parameters.map { case (paramT, _) =>
          val oldparam = paramT.forward.cloneTensor
          val param = NDArray.tensorToNDArray(paramT.forward.value)
          def f(p: NDArray[Double]) = {
            val p2 = NDArray.tensorFromNDArray(p, if (cuda) CudaDevice(0) else lamp.CPU)
            paramT.forward.zero_()
            ATen.add_out(
              paramT.forward.value,
              paramT.forward.value,
              p2,
              1d
            )
            module.forward((d)).sum.forward.toMat.raw(0)
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
          paramT.forward.zero_()
          ATen.add_out(
            paramT.forward.value,
            paramT.forward.value,
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
      device: lamp.Device 
  )(
      m: NDArray[Double],
      inputToA: Constant => A,
      bToVar: B => Variable,
      moduleF: Scope => M with GenericModule[A, B],
      expectedValue: Double
  ) =
    test(
      id + ": gradient is correct",
      (if (device.isInstanceOf[CudaDevice]) List(CudaTest) else Nil): _*
    ) {

      Scope.root { implicit scope =>
        
        
        val d = const(STen.owned(NDArray.tensorFromNDArray(m, device)))

        {
          implicit val fw : BackpropForwardCache = BackpropForwardCache.simple
          val module1 = moduleF(scope)
          val state = module1.state
          val modifiedState = state.map { case (v, _) =>
            v.forward * (-1)
          }
          module1.load(modifiedState)
          (module1.state zip modifiedState).foreach { case ((st1, _), (st2)) =>

            assert(st1.constantValue.toDevice(CPU).toDoubleArray.toList ==
            st2.toDevice(CPU).toDoubleArray.toList)
          }
        }
        implicit val fw : BackpropForwardCache = BackpropForwardCache.simple
        val module = moduleF(scope)
        val output = bToVar(module.forward(inputToA(d)))
        val sum = output.sum.persist

        val value = sum.forward.copyToDevice(CPU).toDoubleArray(0)
        
        val gradNum = module.parameters.map { case (paramT, _) =>
          val oldparam = paramT.forward.cloneTensor
          val param = NDArray.tensorToNDArray(paramT.forward.value)
          def f(p: NDArray[Double]) = {
            val p2 = NDArray.tensorFromNDArray(p, device)
            paramT.forward.zero_()
            ATen.add_out(
              paramT.forward.value,
              paramT.forward.value,
              p2,
              1d
            )
            bToVar(module.forward(inputToA(d))).sum.forward.copyToDevice(CPU).toDoubleArray(0)
          }
          val eps = 1e-1
          val r = NDArray.zeros(paramT.shape.map(_.toInt)).mapWithIndex {
            case (_, idx) =>
              val epsM = NDArray.zeros(paramT.shape.map(_.toInt))
              epsM.set(idx, eps)
              val a = f(param + epsM)
              val b = f(param - epsM)
              val r = (a - b) / (2 * eps)
              r
          }
          paramT.forward.zero_()
          ATen.add_out(
            paramT.forward.value,
            paramT.forward.value,
            oldparam.value,
            1d
          )
          r
        }
        val gradAuto =
          module.computeGradientsAndDestroyGraph(sum,fw,false).map(_.value).map(NDArray.tensorToNDArray)
        assert(gradAuto.size == gradNum.size)
        gradAuto.zip(gradNum).foreach { case (a, b) =>
          
          assert(
            (a.data.toVec - b.data.toVec).map(math.abs).mean2 < 1e-2
          )
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
      implicit val fw = BackpropForwardCache.simple
      val linear = Linear(3, 1, STenOptions.d)
      val output = linear.forward(const(lamp.saddle.fromMat(mat2x3)))
      val sum = output.sum
      assert(output.forward.sizes.toList == List(2, 1))
      sum.forward
      val grad = linear.computeGradientsAndDestroyGraph(sum,fw,false)
      assert(grad.size == 2)
      ()
    }
  }
  test("linear 3D") {
    Scope.root { implicit scope =>
            implicit val fw = BackpropForwardCache.simple

      val linear = Linear(5, 2, STenOptions.d)
      val output =
        linear.forward(const(STen.ones(List(3, 4, 5), STenOptions.d)))
      val sum = output.sum
      assert(output.forward.sizes.toList == List(3, 4, 2))
      sum.forward
      val grad = linear.computeGradientsAndDestroyGraph(sum,fw,true)
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

  testGradientAndValue("Logistic 1")(
    mat3x2,
    implicit pool =>
      LogisticRegression1(2, 3, const(lamp.saddle.fromMat(mat.ident(3))))(
        pool
      ),
    71
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

  testGradientAndValueND("Conv1D ",  false)(
    nd1x2x3,
    implicit pool =>
      Conv1D(
        param(STen.rand(List(1, 2, 3), STenOptions.d)),
        param(STen.rand(List(1), STenOptions.d)),
        stride = 1,
        padding = 0,
        dilation = 1,
        groups = 1
      ),
    10.7941
  )
  testGradientAndValueND("Conv1D/cuda ",true)(
    nd1x2x3,
    implicit pool =>
      Conv1D(
        param(STen.rand(List(1, 2, 3), STenOptions.d.cudaIndex(0))),
        param(STen.rand(List(1), STenOptions.d.cuda)),
        stride = 1,
        padding = 0,
        dilation = 1,
        groups = 1
      ),
    79.2702
  )
  testGradientAndValueND("Conv2D ",false)(
    nd1x2x3x3,
    implicit pool =>
      Conv2D(
        param(STen.rand(List(1, 2, 3, 3), STenOptions.d)),
        param(STen.rand(List(1), STenOptions.d)),
        stride = 1,
        padding = 0,
        dilation = 1,
        groups = 1
      ),
    100.9049
  )
  testGradientAndValueND("Conv2D/cuda ",true)(
    nd1x2x3x3,
    implicit pool =>
      Conv2D(
        param(STen.rand(List(1, 2, 3, 3), STenOptions.d.cuda)),
        param(STen.rand(List(1), STenOptions.d.cuda)),
        stride = 1,
        padding = 0,
        dilation = 1,
        groups = 1
      ),
    73.9732
  )
 

  // testGradientAndValueNDLong("Embedding ",false)(
  //   nd2x3L,
  //   implicit pool =>
  //     Embedding(
  //       weights = param(STen.rand(List(2, 4), STenOptions.d))
  //     ),
  //   9.3734
  // )
 

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
        Seq(t, t2),
        STen.scalarDouble(2.1, topt),
        false
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
      implicit val fw = ForwardCache.selective
      val mod = sequence(
        Linear(in = 30, out = 3, STenOptions.d),
        Linear(in = 3, out = 2, STenOptions.d),
        Fun(_ => input => input.logSoftMax(dim = 1))
      )

      val parameters = mod.state.map(_._1.forward)

      mod.load(parameters)
      assert(mod.state.map(_._1.forward.toMat) == parameters.map(_.toMat))
      val p2 = parameters.map { t => t * 3 }
      mod.load(p2)

      assert(mod.state.map(_._1.forward.toMat) == parameters.map(_.toMat))
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
    CPU
  )(
    m = nd2x3x2,
    inputToA = v =>
      (
        v,
        Option(
          STen.owned(NDArray.tensorFromLongNDArray(nd2x3L, lamp.CPU))(Scope.free)
        )
      ),
    bToVar = v => v,
    moduleF = implicit pool =>  {
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
  if (aten.Tensor.hasMps()) {
  testGradientAndValueNDGeneric[
    (Variable, Option[STen]),
    Variable,
    TransformerEncoder
  ](
    "transformer encoder MPS ",
    MPS
  )(
    m = nd2x3x2,
    inputToA = v =>
      (
        v,
        Option(
          STen.owned(NDArray.tensorFromLongNDArray(nd2x3L, lamp.MPS))(Scope.free)
        )
      ),
    bToVar = v => v,
    moduleF = implicit pool =>  {
      aten.Tensor.manual_seed(123)
      
      val in = 2
      val attentionHiddenPerHeadDim = 4
      val attentionNumHeads = 5
      val mlpHiddenDim = 7
      val dropout = 0d
      // val padToken = -999L
      val tOpt = MPS.to(STenOptions.f)
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
  }
  testGradientAndValueNDGeneric[
    (Variable, Option[STen]),
    Variable,
    TransformerEncoder
  ](
    "linearized transformer encoder ",
    CPU
  )(
    nd2x3x2,
    v =>
      (
        v,
        Option(
          STen.owned(NDArray.tensorFromLongNDArray(nd2x3L, lamp.CPU))(Scope.free)
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
  val w = param(lamp.saddle.fromMat(mat2x3_2)(pool))

  def forward[S: Sc,F:FW](x: Variable): Variable =
    ((x.mm(w)).logSoftMax(dim = 1).crossEntropy(y).sum + w.sum)

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
      param(STen.ones(List(dim, k), y.eval(pool).options(pool))(pool)),
      Some(param(STen.ones(List(1, k), y.eval(pool).options(pool))(pool)))
    ),
    Fun(_ => input => input.logSoftMax(dim = 1))
  )

  def forward[S: Sc,F:FW](x: Variable): Variable =
    mod.forward(x).crossEntropy(y).sum +
      mod.parameters
        .map(_._1.sum)
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
      param(STen.ones(List(dim, 32), y.eval(pool).options(pool))(pool)),
      Some(param(STen.ones(List(1, 32), y.eval(pool).options(pool))(pool)))
    ),
    Fun(_ => input => input.logSoftMax(dim = 1)),
    Fun(_ => input => input.gelu),
    Linear(
      param(STen.ones(List(32, k), y.eval(pool).options(pool))(pool)),
      Some(param(STen.ones(List(1, k), y.eval(pool).options(pool))(pool)))
    )
  )

  def forward[S: Sc,F:FW](x: Variable): Variable =
    mod.forward(x).crossEntropy(y).sum +
      mod.parameters
        .map(_._1.sum)
        .reduce(_ + _)

  override def state =
    Nil

}
object Mlp1 {
  implicit val load: Load[Mlp1] = Load.identity[Mlp1]
}
