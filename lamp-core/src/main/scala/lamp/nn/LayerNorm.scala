package lamp.nn

import lamp.autograd.{Variable, Constant, param}
import lamp.Sc
import lamp.STen
import lamp.STenOptions

case class LayerNorm(
    scale: Constant,
    bias: Constant,
    eps: Double,
    normalizedDim: List[Int]
) extends Module {

  override val state = List(
    scale -> LayerNorm.Scale,
    bias -> LayerNorm.Bias
  )

  override def forward[S: Sc](x: Variable): Variable =
    x.normalize(normalizedDim, eps) * scale + bias

}

object LayerNorm {
  implicit val trainingMode = TrainingMode.identity[LayerNorm]
  implicit val load = Load.make[LayerNorm](m =>
    tensors => {
      m.scale.value.copyFrom(tensors.head)
      m.bias.value.copyFrom(tensors(1))

    }
  )
  case object Scale extends LeafTag
  case object Bias extends LeafTag
  def apply[S: Sc](
      normalizedDimensions: List[Int],
      tOpt: STenOptions,
      eps: Double = 1e-5
  ): LayerNorm = LayerNorm(
    scale = param(STen.ones(List(1), tOpt)),
    bias = param(STen.zeros(List(1), tOpt)),
    eps = eps,
    normalizedDim = normalizedDimensions
  )
}
