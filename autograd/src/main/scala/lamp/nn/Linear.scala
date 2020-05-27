package lamp.nn

import lamp.autograd.{Variable, param}
import aten.{ATen, TensorOptions}
import lamp.autograd.TensorHelpers

case class Linear(weights: Variable, bias: Option[Variable]) extends Module {

  val parameters = List(
    weights
  ) ++ bias.toList

  def forward(x: Variable): Variable = {
    val v = x.mm(weights.t)
    bias.map(_ + v).getOrElse(v)

  }
}

object Linear {
  def apply(in: Int, out: Int, bias: Boolean = true): Linear =
    Linear(
      weights = param(
        ATen.rand(Array(out, in), TensorOptions.dtypeDouble)
      ),
      bias =
        if (bias)
          Some(param(ATen.rand(Array(1, out), TensorOptions.dtypeDouble)))
        else None
    )
}
