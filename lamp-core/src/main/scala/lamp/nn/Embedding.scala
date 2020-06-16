package lamp.nn

import lamp.autograd.{Variable, param}
import aten.{ATen, TensorOptions}
import lamp.autograd.TensorHelpers
import aten.Tensor

/**
  * Learnable mapping from classes to dense vectors.
  * Equivalent to L * W where
  *   L is the n x C one-hot encoded matrix of the classes
  *   * is matrix multiplication
  *   W is the C x dim dense matrix.
  * W is learnable.
  * L is never computed directly.
  * C is the number of classes.
  * n is the size of the batch.
  *
  * Input is a long tensor with values in [0,C-1].
  */
case class Embedding(weights: Variable) extends Module {
  override def load(parameters: Seq[Tensor]) = {
    val w = param(parameters.head)
    copy(weights = w)
  }
  override val state = List(
    weights -> Embedding.Weights
  )

  def forward(x: Variable): Variable =
    lamp.autograd.Embedding(x, weights).value

}

object Embedding {
  case object Weights extends LeafTag
  def apply(
      classes: Int,
      dimensions: Int,
      tOpt: TensorOptions
  ): Embedding =
    Embedding(
      weights = param(
        ATen.normal_3(
          0d,
          math.sqrt(2d / (classes + dimensions)),
          Array(classes, dimensions),
          tOpt
        )
      )
    )
}
