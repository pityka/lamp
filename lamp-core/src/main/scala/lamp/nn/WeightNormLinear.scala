package lamp.nn

import lamp.Sc
import lamp.autograd.{Variable, Constant, param, GraphConfiguration, GC}
import lamp.scope
import lamp.STen
import lamp.STenOptions

case class WeightNormLinear(
    weightsV: Constant,
    weightsG: Constant,
    bias: Option[Constant],
    conf: GraphConfiguration
) extends Module {

  override val state = List(
    weightsV -> WeightNormLinear.WeightsV,
    weightsG -> WeightNormLinear.WeightsG
  ) ++ bias.toList.map(b => (b, WeightNormLinear.Bias))

  def forward[S: Sc](x: Variable): Variable = {
    implicit val _conf = conf
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
  def apply[S: Sc, G: GC](
      in: Int,
      out: Int,
      tOpt: STenOptions,
      bias: Boolean = true
  ): WeightNormLinear =
    WeightNormLinear(
      weightsV = param(
        STen.normal(0d, math.sqrt(2d / (in + out)), List(out, in), tOpt)
      ),
      weightsG = param(
        STen.normal(0d, 0.01, List(1, in), tOpt)
      ),
      bias =
        if (bias)
          Some(param(STen.zeros(List(1, out), tOpt)))
        else None,
      conf = implicitly[GraphConfiguration]
    )
}
