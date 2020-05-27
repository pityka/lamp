package lamp.nn

import org.saddle._
import org.saddle.ops.BinOps._
import org.saddle.linalg._
import org.scalatest.funsuite.AnyFunSuite
import aten.ATen
import lamp.autograd._
import aten.TensorOptions

class NNSuite extends AnyFunSuite {

  def testGradientAndValue(
      id: String
  )(m: Mat[Double], module: Module, expectedValue: Double) =
    test(id + ": gradient is correct") {
      val d = const(TensorHelpers.fromMat(m))
      val output = module.forward(d)
      val sum = output.sum
      val value = TensorHelpers.toMat(sum.value).raw(0)
      val gradAuto = module.gradients(sum).map(TensorHelpers.toMat)
      val gradNum = module.parameters.map { paramT =>
        val oldparam = ATen.clone(paramT.value)
        val param = TensorHelpers.toMat(paramT.value)
        def f(p: Mat[Double]) = {
          val p2 = TensorHelpers.fromMat(p)
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

  val mat2x3 = Mat(Vec(1d, 2d), Vec(3d, 4d), Vec(5d, 6d))
  val mat3x2 = mat2x3.T

  test("linear") {
    val linear = Linear(3, 1)
    val output = linear.forward(const(TensorHelpers.fromMat(mat2x3)))
    val sum = output.sum
    val grad = linear.gradients(sum)
    assert(output.value.sizes.toList == List(2, 1))
    assert(grad.size == 2)

  }
  testGradientAndValue("Linear 0")(
    mat2x3,
    Linear(
      param(ATen.ones(Array(1, 3), TensorOptions.dtypeDouble)),
      Some(param(ATen.ones(Array(1), TensorOptions.dtypeDouble)))
    ),
    23d
  )
  testGradientAndValue("Logistic 1")(
    mat3x2,
    LogisticRegression1(2, 3, const(TensorHelpers.fromMat(mat.ident(3)))),
    151.0000008318073
  )
  testGradientAndValue("Logistic 2")(
    mat3x2,
    LogisticRegression2(2, 3, const(TensorHelpers.fromMat(mat.ident(3)))),
    12.295836866004327
  )

}

case class LogisticRegression1(dim: Int, k: Int, y: Variable) extends Module {

  val mat2x3_2 = Mat(Vec(-1d, 2d), Vec(3d, -4d), Vec(5d, 6d))
  val w = param(TensorHelpers.fromMat(mat2x3_2))

  def forward(x: Variable): Variable =
    ((x.mm(w)).logSoftMax.crossEntropy(y).sum + w.squaredFrobenius)

  def parameters: Seq[Variable] =
    List(w)

}
case class LogisticRegression2(dim: Int, k: Int, y: Variable) extends Module {

  val mod = Sequential(
    Linear(
      param(ATen.ones(Array(k, dim), TensorOptions.dtypeDouble)),
      Some(param(ATen.ones(Array(1, k), TensorOptions.dtypeDouble)))
    ),
    (v: Variable) => v.logSoftMax
  )

  def forward(x: Variable): Variable =
    mod.forward(x).crossEntropy(y).sum +
      mod.parameters
        .map(_.squaredFrobenius)
        .reduce(_ + _)

  def parameters: Seq[Variable] =
    mod.parameters

}
