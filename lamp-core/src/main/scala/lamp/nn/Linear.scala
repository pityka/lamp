package lamp.nn

import lamp.autograd.{Variable, param}
import aten.{ATen, TensorOptions}
import lamp.autograd.AllocatedVariablePool
case class Linear(weights: Variable, bias: Option[Variable]) extends Module {

  override val state = List(
    weights -> Linear.Weights
  ) ++ bias.toList.map(b => (b, Linear.Bias))

  def forward(x: Variable): Variable = {
    val v = x.mm(weights.t)
    bias.map(_ + v).getOrElse(v)

  }
}

object Linear {
  implicit val trainingMode = TrainingMode.identity[Linear]
  implicit val load = Load.make[Linear] { m => parameters =>
    implicit val pool = m.weights.pool
    val w = param(parameters.head)
    val b = if (m.bias.isDefined) Some(param(parameters(1))) else None
    m.copy(weights = w, bias = b)
  }
  case object Weights extends LeafTag
  case object Bias extends LeafTag
  def apply(
      in: Int,
      out: Int,
      tOpt: TensorOptions,
      bias: Boolean = true
  )(implicit pool: AllocatedVariablePool): Linear =
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
