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

  // test1("mnist tabular mlp") { cuda =>
  //   implicit val pool = new AllocatedVariablePool
  //   val data = org.saddle.csv.CsvParser
  //     .parseSourceWithHeader[Double](
  //       scala.io.Source
  //         .fromInputStream(
  //           new java.util.zip.GZIPInputStream(
  //             getClass.getResourceAsStream("/mnist_test.csv.gz")
  //           )
  //         )
  //     )
  //     .right
  //     .get
  //   val target = ATen.squeeze_0(
  //     TensorHelpers.fromLongMat(
  //       Mat(data.firstCol("label").toVec.map(_.toLong)),
  //       cuda
  //     )
  //   )
  //   val x =
  //     const(TensorHelpers.fromMat(data.filterIx(_ != "label").toMat, cuda))

  //   val tOpt = if (cuda) TensorOptions.d.cuda else TensorOptions.d.cpu
  //   val model = Seq2(
  //     lamp.nn.MLP(784, 10, List(4, 4), tOpt),
  //     Fun(_.logSoftMax(dim = 1))
  //   )

  //   println(model.learnableParameters)
  //   // model.m1.setData(x)

  //   val optim = AdamW(
  //     model.parameters.map(v => (v._1.value, v._2)),
  //     learningRate = simple(0.001),
  //     weightDecay = simple(0.0d)
  //   )

  //   var lastAccuracy = 0d
  //   var lastLoss = 1000000d
  //   var i = 0
  //   while (i < 500) {
  //     val output = model.forward(x)
  //     val prediction = {
  //       val argm = ATen.argmax(output.value, 1, false)
  //       val r = TensorHelpers.toLongMat(argm).toVec
  //       argm.release
  //       r
  //     }
  //     val correct = prediction.zipMap(data.firstCol("label").toVec)((a, b) =>
  //       if (a == b) 1d else 0d
  //     )
  //     val classWeights = ATen.ones(Array(10), x.options)
  //     val loss: Variable = output.nllLoss(target, 10, classWeights)
  //     lastAccuracy = correct.mean2
  //     lastLoss = TensorHelpers.toMat(loss.value).raw(0)
  //     println((i, lastLoss, lastAccuracy))
  //     val gradients = model.gradients(loss)
  //     optim.step(gradients)
  //     i += 1
  //   }
  //   println(lastAccuracy)
  //   assert(lastAccuracy > 0.6)
  //   assert(lastLoss < 100d)

  // }
  test1("mnist tabular") { cuda =>
    implicit val pool = new AllocatedVariablePool
    val data = org.saddle.csv.CsvParser
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
    val target = ATen.squeeze_0(
      TensorHelpers.fromLongMat(
        Mat(data.firstCol("label").toVec.map(_.toLong)),
        cuda
      )
    )
    val x =
      const(TensorHelpers.fromMat(data.filterIx(_ != "label").toMat, cuda))

    val tOpt = if (cuda) TensorOptions.d.cuda else TensorOptions.d.cpu
    val model = Seq2(
      Nnrf.apply(
        levels = 7,
        numFeatures = 32,
        totalDataFeatures = 784,
        out = 10,
        tOpt = tOpt
      ),
      Fun(_.logSoftMax(dim = 1))
    )

    println(model.learnableParameters)
    model.m1.setData(x)

    val optim = AdamW(
      model.parameters.map(v => (v._1.value, v._2)),
      learningRate = simple(0.001),
      weightDecay = simple(0.0d)
    )

    var lastAccuracy = 0d
    var lastLoss = 1000000d
    var i = 0
    while (i < 500) {
      val output = model.forward(x)
      val prediction = {
        val argm = ATen.argmax(output.value, 1, false)
        val r = TensorHelpers.toLongMat(argm).toVec
        argm.release
        r
      }
      val correct = prediction.zipMap(data.firstCol("label").toVec)((a, b) =>
        if (a == b) 1d else 0d
      )
      val classWeights = ATen.ones(Array(10), x.options)
      val loss: Variable = output.nllLoss(target, 10, classWeights)
      lastAccuracy = correct.mean2
      lastLoss = TensorHelpers.toMat(loss.value).raw(0)
      println((i, lastLoss, lastAccuracy))
      val gradients = model.gradients(loss)
      optim.step(gradients)
      i += 1
    }
    println(lastAccuracy)
    assert(lastAccuracy > 0.6)
    assert(lastLoss < 100d)

  }
}
