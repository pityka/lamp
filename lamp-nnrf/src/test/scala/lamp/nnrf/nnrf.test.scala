package lamp.nnrf

import org.saddle._
import org.saddle.ops.BinOps._
import org.saddle.linalg._
import org.scalatest.funsuite.AnyFunSuite
import lamp._
import lamp.util.NDArray
import lamp.nn._
import lamp.autograd._
import aten.ATen
import aten.TensorOptions

class NnrfSuite extends AnyFunSuite {
  val cpuPool = new AllocatedVariablePool
  val cudaPool = new AllocatedVariablePool
  def selectPool(cuda: Boolean) = if (cuda) cudaPool else cpuPool
  def testGradient[M <: Module: Load](
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
      println(output.toMat)
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

    }

  val mat2x3 = Mat(Vec(-1d, 2d), Vec(3d, -4d), Vec(5d, 6d))

  // testGradient("gradients")(
  //   mat2x3,
  //   implicit pool => Nnrf(3, 2, 3, TensorOptions.dtypeDouble())
  // )
  // testGradient("gradients cuda", cuda = true)(
  //   mat2x3,
  //   implicit pool => Nnrf(3, 2, 3, TensorOptions.dtypeDouble().cuda())
  // )

  def test1(id: String)(fun: Boolean => Unit) = {
    test(id) { fun(false) }
    test(id + "/CUDA", CudaTest) { fun(true) }
  }

  test1("mnist tabular") { cuda =>
    implicit val pool = new AllocatedVariablePool
    val dataTrain = org.saddle.csv.CsvParser
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
    val target = dataTrain.firstCol("label").toVec.map(_.toLong)
    val device = if (cuda) CudaDevice(0) else CPU
    val precision = SinglePrecision
    val x = dataTrain.filterIx(_ != "label").toMat

    val tOpt = if (cuda) TensorOptions.f.cuda else TensorOptions.f.cpu
    val (trained, lastLoss) = Nnrf.trainClassification(
      features = x,
      target = target,
      device = device,
      precision = precision,
      levels = 7,
      numClasses = 10,
      classWeights = vec.ones(10),
      epochs = 100,
      logger = Some(scribe.Logger("test"))
    )

    val output = Nnrf.predict(dataTest.filterIx(_ != "label").toMat, trained)
    val prediction = output.reduceRows((v, _) => v.argmax)

    val correct = prediction.zipMap(dataTest.firstCol("label").toVec)((a, b) =>
      if (a == b) 1d else 0d
    )

    val accuracy = correct.mean2
    println(accuracy)
    assert(accuracy > 0.6)
    assert(lastLoss < 0.6)

  }
}
