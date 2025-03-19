package lamp.nn

import lamp.autograd.{Variable, Constant, param}
import lamp.Sc
import lamp.STen
import lamp.STenOptions

/** Inputs of size (sequence length * batch * in dim) Outputs of size (sequence
  * length * batch * output dim) Applies a linear function to each time step
  */
case class SeqLinear(
    weight: Constant,
    bias: Constant
) extends Module {

  override def state =
    List(
      (weight, SeqLinear.Weight),
      (bias, SeqLinear.Bias)
    )

  override def forward[S: Sc](x: Variable) = {
    val timesteps = x.shape.head
    val outputs = (0 until timesteps.toInt).map { t =>
      val xt = x.select(0, t)
      (xt.mm(weight) + bias)
    }
    Variable.concatenateAddNewDim(outputs)

  }

}

object SeqLinear {
  implicit val trainingMode: TrainingMode[SeqLinear] =
    TrainingMode.identity[SeqLinear]
  implicit val load: Load[SeqLinear] = Load.make[SeqLinear] { m => tensors =>
    m.weight.value.copyFrom(tensors(0))
    m.bias.value.copyFrom(tensors(1))

  }
  case object Weight extends LeafTag
  case object Bias extends LeafTag

  def apply[S: Sc](
      in: Int,
      out: Int,
      tOpt: STenOptions
  ): SeqLinear =
    SeqLinear(
      weight = param(
        STen.normal(
          0d,
          math.sqrt(2d / (in + out)),
          List(in, out),
          tOpt
        )
      ),
      bias = param(
        STen.zeros(
          List(1, out),
          tOpt
        )
      )
    )

}
