package lamp.nn

import lamp.autograd.{Variable, param, const}
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
case class LinearSeq(
    weight: Variable,
    bias: Variable,
    dropout: Double,
    train: Boolean
) extends Module {
  override def asEval: LinearSeq = copy(train = false)
  override def asTraining: LinearSeq = copy(train = true)

  override def load(tensors: Seq[Tensor]): LinearSeq = copy(
    weight = param(tensors(0)),
    bias = param(tensors(1))
  )

  override def state: Seq[(Variable, PTag)] =
    List(
      (weight, LinearSeq.Weight),
      (bias, LinearSeq.Bias)
    )

  override def forward(x: Variable) = {
    val timesteps = x.shape.head
    val outputs = (0 until timesteps.toInt).map { t =>
      val xt = x.select(0, t)
      (xt.mm(weight) + bias).relu.dropout(dropout, train)
    }
    ConcatenateAddNewDim(outputs).value

  }

}

object LinearSeq {
  case object Weight extends LeafTag
  case object Bias extends LeafTag

  def apply(
      in: Int,
      out: Int,
      dropout: Double,
      tOpt: TensorOptions
  ): LinearSeq =
    LinearSeq(
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
      ),
      dropout = dropout,
      train = true
    )

}
