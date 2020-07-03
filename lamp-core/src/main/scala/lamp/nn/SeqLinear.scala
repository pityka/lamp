package lamp.nn

import lamp.autograd.{Variable, param, const}
import lamp.autograd.AllocatedVariablePool
import aten.Tensor
import aten.ATen
import scala.collection.mutable
import lamp.autograd.ConcatenateAddNewDim
import aten.TensorOptions
import cats.effect.concurrent.Ref
import cats.effect.IO

/** Inputs of size (sequence length * batch * in dim)
  * Outputs of size (sequence length * batch * output dim)
  * Applies a linear function to each time step
  */
case class SeqLinear(
    weight: Variable,
    bias: Variable
) extends Module {

  override def state: Seq[(Variable, PTag)] =
    List(
      (weight, SeqLinear.Weight),
      (bias, SeqLinear.Bias)
    )

  override def forward(x: Variable) = {
    val timesteps = x.shape.head
    val outputs = (0 until timesteps.toInt).map { t =>
      val xt = x.select(0, t)
      (xt.mm(weight) + bias)
    }
    ConcatenateAddNewDim(outputs).value

  }

}

object SeqLinear {
  implicit val trainingMode = TrainingMode.identity[SeqLinear]
  implicit val load = Load.make[SeqLinear] { m => tensors =>
    implicit val pool = m.weight.pool
    m.copy(
      weight = param(tensors(0)),
      bias = param(tensors(1))
    )
  }
  case object Weight extends LeafTag
  case object Bias extends LeafTag

  def apply(
      in: Int,
      out: Int,
      tOpt: TensorOptions
  )(implicit pool: AllocatedVariablePool): SeqLinear =
    SeqLinear(
      weight = param(
        ATen.normal_3(
          0d,
          math.sqrt(2d / (in + out)),
          Array(in, out),
          tOpt
        )
      ),
      bias = param(
        ATen.zeros(
          Array(1, out),
          tOpt
        )
      )
    )

}
