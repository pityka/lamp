package lamp.nn

import lamp.autograd.{Variable, param}
import aten.{ATen, TensorOptions}
import lamp.autograd.TensorHelpers

case class WeightNormLinear(
    weightsV: Variable,
    weightsG: Variable,
    bias: Option[Variable]
) extends Module {

  val parameters = List(
    weightsV -> WeightNormLinear.WeightsV,
    weightsG -> WeightNormLinear.WeightsG
  ) ++ bias.toList.map(b => (b, WeightNormLinear.Bias))

  def forward(x: Variable): Variable = {
    val weights = lamp.autograd.WeightNorm(weightsV, weightsG, 0).value
    val v = x.mm(weights.t)
    bias.map(_ + v).getOrElse(v)

  }
}

object WeightNormLinear {
  case object WeightsV extends LeafTag
  case object WeightsG extends LeafTag
  case object Bias extends LeafTag
  def apply(
      in: Int,
      out: Int,
      bias: Boolean = true,
      tOpt: TensorOptions = TensorOptions.dtypeDouble
  ): WeightNormLinear =
    WeightNormLinear(
      weightsV = param(
        ATen.rand(Array(out, in), tOpt)
      ),
      weightsG = param(
        ATen.rand(Array(1, in), tOpt)
      ),
      bias =
        if (bias)
          Some(param(ATen.rand(Array(1, out), tOpt)))
        else None
    )
}
