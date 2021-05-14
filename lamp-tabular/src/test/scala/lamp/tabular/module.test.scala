package lamp.tabular

import org.saddle._
import org.saddle.ops.BinOps._
import org.scalatest.funsuite.AnyFunSuite
import aten.ATen
import lamp.autograd._
import lamp.util.NDArray
import aten.Tensor
import lamp.nn._
import lamp.SinglePrecision
import lamp.CPU
import lamp.CudaDevice
import lamp.Device
import lamp.DoublePrecision
import java.io.File
import lamp.Scope
import lamp.STen
import lamp.STenOptions
import cats.effect.unsafe.implicits.global
class TabularResidualModuleSuite extends AnyFunSuite {

  def test1(id: String)(fun: Boolean => Unit) = {
    test(id) { fun(false) }
    test(id + "/CUDA", CudaTest) { fun(true) }
  }
  def testGradientAndValue[M <: Module: Load](
      id: String,
      cuda: Boolean = false
  )(
      m: Mat[Double],
      moduleF: Scope => M with Module
  ) =
    test(
      id + ": gradient is correct",
      (if (cuda) List(CudaTest) else Nil): _*
    ) {

      Scope.root { implicit scope =>
        val d = const(STen.fromMat(m, cuda))
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
        val gradAuto = module.gradients(sum).map(_.get).map(_.toMat)
        val gradNum = module.parameters.map { case (paramT, _) =>
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
        ()
      }
    }

  val mat2x3 = Mat(Vec(1d, 2d), Vec(3d, 4d), Vec(5d, 6d))
  testGradientAndValue("tabular residual block")(
    mat2x3,
    implicit scope => TabularResidual.make(3, 16, 2, STenOptions.d, 0d)
  )

  test("tabular embedding") {
    Scope.root { implicit scope =>
      val mat2x1L = Mat(Vec(1L, 2L))
      val mod =
        TabularEmbedding.make(List(3 -> 2), STenOptions.d)

      val result = mod
        .forward(
          (
            List(const(STen.fromLongMat(mat2x1L, false))),
            const(STen.fromMat(mat2x3, false))
          )
        )
        .toMat
      assert(result.numRows == 2)
      assert(result.numCols == 5)
      ()
    }

  }
  test("mnist") {
    Scope.root { implicit scope =>
      def train(
          features: STen,
          target: STen,
          dataLayout: Seq[Metadata],
          targetType: TargetType,
          device: Device
      ) = {
        val precision =
          if (features.options.isDouble) DoublePrecision
          else if (features.options.isFloat) SinglePrecision
          else throw new RuntimeException("Expected float or double tensor")
        val numInstances = features.sizes.apply(0).toInt
        val rng = org.saddle.spire.random.rng.Cmwc5.apply()
        val minibatchSize =
          if (numInstances < 1024) 8 else if (numInstances < 4096) 64 else 256
        val cvFolds =
          AutoLoop.makeCVFolds(
            numInstances,
            k = 2,
            1,
            rng
          )
        val ensembleFolds =
          AutoLoop
            .makeCVFolds(numInstances, k = 2, 1, rng)
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
          knnK = Nil,
          extratreesK = Nil,
          extratreesM = Seq(5),
          extratreesNMin = Seq(2),
          extratreeParallelism = 1,
          device = device,
          precision = precision,
          minibatchSize = minibatchSize,
          logger = None,
          ensembleFolds = ensembleFolds,
          learningRate = 0.0001,
          knnMinibatchSize = 512,
          rng = rng
        )(scope)
      }

      val data = org.saddle.csv.CsvParser
        .parseSourceWithHeader[Double](
          scala.io.Source
            .fromInputStream(
              new java.util.zip.GZIPInputStream(
                getClass.getResourceAsStream("/mnist_train.csv.gz")
              )
            )
        )
        .toOption
        .get

      val target =
        STen
          .fromLongMat(
            Mat(data.firstCol("label").toVec.map(_.toLong)),
            false
          )
          .squeeze

      val features =
        STen
          .fromMat(
            data.filterIx(_ != "label").toMat,
            false
          )
          .copyTo(STenOptions.d)

      val device = if (Tensor.cudnnAvailable()) CudaDevice(0) else CPU

      val trained = train(
        features,
        target,
        0 until features.sizes.apply(1).toInt map (_ => Numerical),
        Classification(10, vec.ones(10).toSeq),
        device
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
        .toOption
        .get
      val targetTest = dataTest.firstCol("label").toVec.map(_.toLong)
      val featuresTest =
        STen
          .fromMat(
            dataTest.filterIx(_ != "label").toMat,
            false
          )
          .copyTo(STenOptions.d)

      val savePath = File.createTempFile("lampsave", "data").getAbsolutePath
      Serialization.saveModel(trained, savePath)
      val trained2 = Serialization.loadModel(savePath)
      trained2
        .predict(featuresTest)
        .map { modelOutput =>
          val prediction = {
            val t = modelOutput.argmax(1, false)
            val r = t.toLongMat.toVec
            r
          }
          val corrects =
            prediction.zipMap(targetTest)((a, b) => if (a == b) 1d else 0d)
          println(
            s" corrects: ${corrects.mean}"
          )

        }
        .unsafeRunSync()
    }
  }
}
