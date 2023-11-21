package lamp.nn

import lamp.autograd.{Variable, Constant, param}
import lamp.Sc
import lamp.STen
import lamp.STenOptions
case class Linear(weights: Constant, bias: Option[Constant]) extends Module {

  override val state = List(
    weights -> Linear.Weights
  ) ++ bias.toList.map(b => (b, Linear.Bias))

  private def mm1[S: Sc](a: Variable, b: Variable) = {
    val shape = a.shape
    a.view(List(-1, shape.last)).mm(b).view(shape.dropRight(1) :+ -1L)
  }

  def forward[S: Sc](x: Variable): Variable = {
    val v =
      if (x.shape.size == 2 && weights.shape.size == 2)
        x.mm(weights)
      else if (weights.shape.size == 3) {
        x
          .view(List(x.shape(0), weights.shape(0), -1L))
          .transpose(0, 1)
          .bmm(
            weights
          )
          .transpose(0, 1)
          .reshape(List(x.shape(0), -1L))
      } else mm1(x, weights)

    bias.map(_ + v).getOrElse(v)
  }
}

object Linear {
  implicit val trainingMode : TrainingMode[Linear] = TrainingMode.identity[Linear]
  implicit val load : Load[Linear] = Load.make[Linear] { m => parameters =>
    m.weights.value.copyFrom(parameters.head)
    m.bias.foreach(_.value.copyFrom(parameters(1)))
  }
  case object Weights extends LeafTag
  case object Bias extends LeafTag
  def apply[S: Sc](
      in: Int,
      out: Int,
      tOpt: STenOptions,
      bias: Boolean = true,
      numHeads: Int = 1
  ): Linear =
    Linear(
      weights = param(
        STen.normal(
          0d,
          math.sqrt(2d / (in + out)),
          if (numHeads == 1) List(in, out)
          else List(numHeads, in / numHeads, out / numHeads),
          tOpt
        )
      ),
      bias =
        if (bias)
          Some(param(STen.zeros(List(1, out), tOpt)))
        else None
    )
}
