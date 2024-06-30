package lamp.nn

import lamp.autograd.{Variable, Constant, param}
import lamp.{Sc, Scope}
import lamp.STen
import lamp.STenOptions

case class LayerNorm(
    scale: Option[Constant],
    bias: Option[Constant],
    eps: Double,
    normalizedShape: List[Long]
) extends Module {

  override val state =
    scale.map(_ -> LayerNorm.Scale).toList ++
      bias.map(_ -> LayerNorm.Bias).toList

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
      val a = (m.scale.toList ++ m.bias.toList)
      a.zip(tensors.take(a.length)).foreach { case (a, b) =>
        a.value.copyFrom(b)
      }

    }
  )
  case object Scale extends LeafTag
  case object Bias extends LeafTag
  def apply[S: Sc](
      normalizedShape: List[Long],
      tOpt: STenOptions,
      eps: Double = 1e-5,
      scale: Boolean = false,
      bias: Boolean = false
  ): LayerNorm = LayerNorm(
    scale = if (scale) Some(param(STen.ones(normalizedShape, tOpt))) else None,
    bias = if (bias) Some(param(STen.zeros(normalizedShape, tOpt))) else None,
    eps = eps,
    normalizedShape = normalizedShape
  )
}
