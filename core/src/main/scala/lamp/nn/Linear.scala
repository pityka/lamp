package lamp.nn

import lamp.autograd.{Variable, param}
import aten.{ATen, TensorOptions}
import lamp.autograd.TensorHelpers
import aten.Tensor

case class Linear(weights: Variable, bias: Option[Variable]) extends Module {
  def load(parameters: Seq[Tensor]) = {
    val w = param(parameters.head)
    val b = if (bias.isDefined) Some(param(parameters(1))) else None
    copy(weights = w, bias = b)
  }
  val parameters = List(
    weights -> Linear.Weights
  ) ++ bias.toList.map(b => (b, Linear.Bias))

  def forward(x: Variable): Variable = {
    val v = x.mm(weights.t)
    bias.map(_ + v).getOrElse(v)

  }
}

object Linear {
  case object Weights extends LeafTag
  case object Bias extends LeafTag
  def apply(
      in: Int,
      out: Int,
      bias: Boolean = true,
      tOpt: TensorOptions = TensorOptions.dtypeDouble
  ): Linear =
    Linear(
      weights = param(
        ATen.normal_3(0d, math.sqrt(2d / (in + out)), Array(out, in), tOpt)
      ),
      bias =
        if (bias)
          Some(param(ATen.zeros(Array(1, out), tOpt)))
        else None
    )
}
