package lamp.nn

import lamp.autograd.{Variable, Constant, param}
import lamp.STenOptions
import lamp.Sc
import lamp.scope
import lamp.STen

/** Learnable mapping from classes to dense vectors. Equivalent to L * W where L
  * is the n x C one-hot encoded matrix of the classes * is matrix
  * multiplication W is the C x dim dense matrix. W is learnable. L is never
  * computed directly. C is the number of classes. n is the size of the batch.
  *
  * Input is a long tensor with values in [0,C-1]. Input shape is arbitrary,
  * (*). Output shape is (* x D) where D is the embedding dimension.
  */
case class Embedding(weights: Constant) extends Module {
  val state = List(
    weights -> Embedding.Weights
  )

  def forward[S: Sc](x: Variable): Variable =
    new lamp.autograd.Embedding(scope, x, weights).value

}

object Embedding {
  implicit val trainingMode: TrainingMode[Embedding] =
    TrainingMode.identity[Embedding]
  implicit val load: Load[Embedding] = Load.make[Embedding] { m => parameters =>
    m.weights.value.copyFrom(parameters.head)
  }
  case object Weights extends LeafTag
  def apply[S: Sc](
      classes: Int,
      dimensions: Int,
      tOpt: STenOptions
  ): Embedding =
    Embedding(
      weights = param(
        STen.normal(
          0d,
          math.sqrt(2d / (classes + dimensions)),
          List(classes, dimensions),
          tOpt
        )
      )
    )
}
