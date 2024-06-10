package lamp.nn

import lamp.autograd.{Variable, Constant, param}
import lamp.{Sc, Scope}
import lamp.STen
import lamp.STenOptions

case class LayerNorm(
    scale: Constant,
    bias: Constant,
    eps: Double,
    normalizedShape: List[Long]
) extends Module {

  override val state = List(
    scale -> LayerNorm.Scale,
    bias -> LayerNorm.Bias
  )

  override def forward[S: Sc](x: Variable): Variable =
    (new lamp.autograd.LayerNormOp(
      implicitly[Scope],
      x,
      scale,
      bias,
      normalizedShape,
      eps
    )).value

}

object LayerNorm {
  implicit val trainingMode: TrainingMode[LayerNorm] =
    TrainingMode.identity[LayerNorm]
  implicit val load: Load[LayerNorm] = Load.make[LayerNorm](m =>
    tensors => {
      m.scale.value.copyFrom(tensors.head)
      m.bias.value.copyFrom(tensors(1))

    }
  )
  case object Scale extends LeafTag
  case object Bias extends LeafTag
  def apply[S: Sc](
      normalizedShape: List[Long],
      tOpt: STenOptions,
      eps: Double = 1e-5
  ): LayerNorm = LayerNorm(
    scale = param(STen.ones(normalizedShape, tOpt)),
    bias = param(STen.zeros(normalizedShape, tOpt)),
    eps = eps,
    normalizedShape = normalizedShape
  )
}
