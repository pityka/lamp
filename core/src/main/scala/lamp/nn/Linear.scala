package lamp.nn

import lamp.autograd.{Variable, param}
import aten.{ATen, TensorOptions}
import lamp.autograd.TensorHelpers

case class Linear(weights: Variable, bias: Option[Variable]) extends Module {

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
        ATen.rand(Array(out, in), tOpt)
      ),
      bias =
        if (bias)
          Some(param(ATen.rand(Array(1, out), tOpt)))
        else None
    )
}
