package lamp.nn

import lamp.autograd.{Variable, Constant, param, GraphConfiguration, GC}
import lamp.Sc
import lamp.STen
import lamp.STenOptions
case class Linear(weights: Constant, bias: Option[Constant])(implicit
    conf: GraphConfiguration
) extends Module {

  override val state = List(
    weights -> Linear.Weights
  ) ++ bias.toList.map(b => (b, Linear.Bias))

  def forward[S: Sc](x: Variable): Variable = {
    val v = x.mm(weights)
    bias.map(_ + v).getOrElse(v)

  }
}

object Linear {
  implicit val trainingMode = TrainingMode.identity[Linear]
  implicit val load = Load.make[Linear] { m => parameters =>
    m.weights.value.copyFrom(parameters.head)
    m.bias.foreach(_.value.copyFrom(parameters(1)))
  }
  case object Weights extends LeafTag
  case object Bias extends LeafTag
  def apply[S: Sc, G: GC](
      in: Int,
      out: Int,
      tOpt: STenOptions,
      bias: Boolean = true
  ): Linear =
    Linear(
      weights = param(
        STen.normal(0d, math.sqrt(2d / (in + out)), List(in, out), tOpt)
      ),
      bias =
        if (bias)
          Some(param(STen.zeros(List(1, out), tOpt)))
        else None
    )
}
