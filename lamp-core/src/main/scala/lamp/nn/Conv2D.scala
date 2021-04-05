package lamp.nn

import lamp.autograd.{Variable, Constant, param, Conv2D => Conv2dOp, const}
import lamp.STenOptions
import lamp.Sc
import lamp.scope
import lamp.STen
case class Conv2D(
    weights: Constant,
    bias: Constant,
    stride: Long,
    padding: Long,
    dilation: Long,
    groups: Long
) extends Module {

  override val state = List(
    weights -> Conv2D.Weights,
    bias -> Conv2D.Bias
  )

  def forward[S: Sc](x: Variable): Variable =
    new Conv2dOp(
      scope,
      x,
      weights,
      bias,
      stride,
      padding,
      dilation,
      groups
    ).value

}

object Conv2D {
  implicit val trainingMode = TrainingMode.identity[Conv2D]
  implicit val load = Load.make[Conv2D](m =>
    parameters => {
      m.weights.value.copyFrom(parameters.head)
      m.bias.value.copyFrom(parameters(1))
    }
  )
  case object Weights extends LeafTag
  case object Bias extends LeafTag
  def apply[S: Sc](
      inChannels: Long,
      outChannels: Long,
      kernelSize: Long,
      tOpt: STenOptions,
      bias: Boolean = false,
      stride: Long = 1,
      padding: Long = 0,
      dilation: Long = 1,
      groups: Long = 1
  ): Conv2D = {
    val weightVar = param(
      STen.normal(
        0d,
        math.sqrt(2d / (outChannels + inChannels)),
        List(outChannels, inChannels, kernelSize, kernelSize),
        tOpt
      )
    )
    val biasVar = {
      val t = STen.zeros(List(outChannels), tOpt)
      if (bias) param(t) else const(t)
    }
    Conv2D(
      weightVar,
      biasVar,
      stride,
      padding,
      dilation,
      groups
    )

  }

}
