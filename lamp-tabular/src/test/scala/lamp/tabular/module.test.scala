package lamp.tabular

import org.saddle._
import org.saddle.ops.BinOps._
import org.scalatest.funsuite.AnyFunSuite
import aten.ATen
import lamp.autograd._
import aten.TensorOptions
import lamp.util.NDArray
import aten.Tensor
import cats.effect.IO
import lamp.nn._
import lamp.SinglePrecision
import lamp.CPU
import lamp.CudaDevice
import lamp.Device
import lamp.DoublePrecision
import java.io.File

class TabularResidualModuleSuite extends AnyFunSuite {
  val cpuPool = new AllocatedVariablePool
  val cudaPool = new AllocatedVariablePool
  def selectPool(cuda: Boolean) = if (cuda) cudaPool else cpuPool

  def test1(id: String)(fun: Boolean => Unit) = {
    test(id) { fun(false) }
    test(id + "/CUDA", CudaTest) { fun(true) }
  }
  def testGradientAndValue[M <: Module: Load](
      id: String,
      cuda: Boolean = false
  )(
      m: Mat[Double],
      moduleF: AllocatedVariablePool => M with Module
  ) =
    test(id + ": gradient is correct", (if (cuda) List(CudaTest) else Nil): _*) {
      implicit val pool = selectPool(cuda)
      val d = const(TensorHelpers.fromMat(m, cuda))
      val module = moduleF(pool)

      {
        val module1 = moduleF(pool)
        val state = module1.state
        val modifiedState = state.map {
          case (v, _) =>
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

    }

  val mat2x3 = Mat(Vec(1d, 2d), Vec(3d, 4d), Vec(5d, 6d))
  testGradientAndValue("tabular residual block")(
    mat2x3,
    implicit pool =>
      TabularResidual.make(3, 16, 2, TensorOptions.dtypeDouble, 0d)
  )

  test("tabular embedding") {
    implicit val pool = selectPool(false)
    val mat2x1L = Mat(Vec(1L, 2L))
    val mod =
      TabularEmbedding.make(List(3 -> 2), TensorOptions.dtypeDouble())

    val result = mod
      .forward(
        (
          List(const(TensorHelpers.fromLongMat(mat2x1L, false))),
          const(TensorHelpers.fromMat(mat2x3, false))
        )
      )
      .toMat
    assert(result.numRows == 2)
    assert(result.numCols == 5)

  }
  test("mnist") {

    def train(
        features: Tensor,
        target: Tensor,
        dataLayout: Seq[Metadata],
        targetType: TargetType,
        device: Device,
        logFrequency: Int
    ) = {
      implicit val pool = new AllocatedVariablePool
      val precision =
        if (features.options.isDouble) DoublePrecision
        else if (features.options.isFloat) SinglePrecision
        else throw new RuntimeException("Expected float or double tensor")
      val numInstances = features.sizes.apply(0).toInt

      val minibatchSize =
        if (numInstances < 1024) 8 else if (numInstances < 4096) 64 else 256
      val cvFolds =
        AutoLoop.makeCVFolds(
          numInstances,
          k = 2,
          1
        )
      val ensembleFolds =
        AutoLoop
          .makeCVFolds(numInstances, k = 2, 1)
      AutoLoop.train(
        dataFullbatch = features,
        targetFullbatch = target,
        folds = cvFolds,
        targetType = targetType,
        dataLayout = dataLayout,
        epochs = Seq(4),
        weighDecays = Seq(0.0),
        dropouts = Seq(0.5),
        hiddenSizes = Seq(32),
        knnK = Seq(5),
        extratreesK = Seq(30),
        extratreesM = Seq(5),
        extratreesNMin = Seq(2),
        extratreeParallelism = 1,
        device = device,
        precision = precision,
        minibatchSize = minibatchSize,
        logFrequency = logFrequency,
        logger = None,
        ensembleFolds = ensembleFolds,
        learningRate = 0.0001,
        prescreenHyperparameters = true,
        knnMinibatchSize = 512
      )
    }

    implicit val pool = new AllocatedVariablePool
    val data = org.saddle.csv.CsvParser
      .parseSourceWithHeader[Double](
        scala.io.Source
          .fromInputStream(
            new java.util.zip.GZIPInputStream(
              getClass.getResourceAsStream("/mnist_train.csv.gz")
            )
          )
      )
      .right
      .get

    val target = ATen.squeeze_0(
      TensorHelpers.fromLongMat(
        Mat(data.firstCol("label").toVec.map(_.toLong)),
        false
      )
    )
    val features =
      TensorHelpers
        .fromMat(
          data.filterIx(_ != "label").toMat,
          false
        )
        .to(TensorOptions.dtypeFloat(), true)

    val device = if (Tensor.cudnnAvailable()) CudaDevice(0) else CPU

    val trained = train(
      features,
      target,
      0 until features.sizes.apply(1).toInt map (_ => Numerical),
      Classification(10, vec.ones(10).toSeq),
      device,
      logFrequency = 100
    ).unsafeRunSync()

    val dataTest = org.saddle.csv.CsvParser
      .parseSourceWithHeader[Double](
        scala.io.Source
          .fromInputStream(
            new java.util.zip.GZIPInputStream(
              getClass.getResourceAsStream("/mnist_test.csv.gz")
            )
          )
      )
      .right
      .get
    val targetTest = dataTest.firstCol("label").toVec.map(_.toLong)
    val featuresTest =
      TensorHelpers
        .fromMat(
          dataTest.filterIx(_ != "label").toMat,
          false
        )
        .to(TensorOptions.dtypeFloat(), true)

    val savePath = File.createTempFile("lampsave", "data").getAbsolutePath
    Serialization.saveModel(trained, savePath)
    val trained2 = Serialization.loadModel(savePath).right.get
    trained2
      .predict(featuresTest)
      .use { modelOutput =>
        IO {
          val prediction = {
            val t = ATen.argmax(modelOutput, 1, false)
            val r = TensorHelpers
              .toLongMat(t)
              .toVec
            t.release
            r
          }
          val corrects =
            prediction.zipMap(targetTest)((a, b) => if (a == b) 1d else 0d)
          println(
            s" corrects: ${corrects.mean}"
          )
        }
      }
      .unsafeRunSync()
  }

}
