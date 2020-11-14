package lamp.nn

import lamp.Sc
import lamp.autograd.{Variable, Constant, param}
import aten.{TensorOptions}
import lamp.scope
import lamp.STen

case class WeightNormLinear(
    weightsV: Constant,
    weightsG: Constant,
    bias: Option[Constant]
) extends Module {

  override val state = List(
    weightsV -> WeightNormLinear.WeightsV,
    weightsG -> WeightNormLinear.WeightsG
  ) ++ bias.toList.map(b => (b, WeightNormLinear.Bias))

  def forward[S: Sc](x: Variable): Variable = {
    val weights =
      new lamp.autograd.WeightNorm(scope, weightsV, weightsG, 0).value
    val v = x.mm(weights.t)
    bias.map(_ + v).getOrElse(v)

  }
}

object WeightNormLinear {
  implicit val trainingMode = TrainingMode.identity[WeightNormLinear]
  implicit val load = Load.make[WeightNormLinear] { m => parameters =>
    m.weightsV.value.copyFrom(parameters.head)
    m.weightsG.value.copyFrom(parameters(1))
    m.bias.foreach(_.value.copyFrom(parameters(2)))
  }
  case object WeightsV extends LeafTag
  case object WeightsG extends LeafTag
  case object Bias extends LeafTag
  def apply[S: Sc](
      in: Int,
      out: Int,
      tOpt: TensorOptions,
      bias: Boolean = true
  ): WeightNormLinear =
    WeightNormLinear(
      weightsV = param(
        STen.normal(0d, math.sqrt(2d / (in + out)), Array(out, in), tOpt)
      ),
      weightsG = param(
        STen.normal(0d, 0.01, Array(1, in), tOpt)
      ),
      bias =
        if (bias)
          Some(param(STen.zeros(Array(1, out), tOpt)))
        else None
    )
}
