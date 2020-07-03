package lamp.nn

import lamp.autograd.{Variable, param, AllocatedVariablePool}
import aten.{ATen, TensorOptions}
import lamp.autograd.TensorHelpers
import aten.Tensor

case class WeightNormLinear(
    weightsV: Variable,
    weightsG: Variable,
    bias: Option[Variable]
) extends Module {

  override val state = List(
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
  implicit val trainingMode = TrainingMode.identity[WeightNormLinear]
  implicit val load = Load.make[WeightNormLinear] { m => parameters =>
    implicit val pool = m.weightsV.pool
    val wV = param(parameters.head)
    val wG = param(parameters(1))
    val b = if (m.bias.isDefined) Some(param(parameters(2))) else None
    m.copy(weightsV = wV, weightsG = wG, bias = b)
  }
  case object WeightsV extends LeafTag
  case object WeightsG extends LeafTag
  case object Bias extends LeafTag
  def apply(
      in: Int,
      out: Int,
      tOpt: TensorOptions,
      bias: Boolean = true
  )(implicit pool: AllocatedVariablePool): WeightNormLinear =
    WeightNormLinear(
      weightsV = param(
        ATen.normal_3(0d, math.sqrt(2d / (in + out)), Array(out, in), tOpt)
      ),
      weightsG = param(
        ATen.normal_3(0d, 0.01, Array(1, in), tOpt)
      ),
      bias =
        if (bias)
          Some(param(ATen.zeros(Array(1, out), tOpt)))
        else None
    )
}
