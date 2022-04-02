package lamp.nn

import lamp.autograd.{
  Variable,
  Constant,
  param,
  Conv2DTransposed => Conv2DTOp,
  const
}
import lamp.STenOptions
import lamp.Sc
import lamp.scope
import lamp.STen
case class Conv2DTransposed(
    weights: Constant,
    bias: Constant,
    stride: Long,
    padding: Long,
    dilation: Long
) extends Module {

  override val state = List(
    weights -> Conv2DTransposed.Weights,
    bias -> Conv2DTransposed.Bias
  )

  def forward[S: Sc](x: Variable): Variable =
    new Conv2DTOp(scope, x, weights, bias, stride, padding, dilation).value

}

object Conv2DTransposed {
  implicit val trainingMode : TrainingMode[Conv2DTransposed] = TrainingMode.identity[Conv2DTransposed]
  implicit val load : Load[Conv2DTransposed] = Load.make[Conv2DTransposed](m =>
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
      dilation: Long = 1
  ): Conv2DTransposed = {
    val weightVar = param(
      STen.normal(
        0d,
        math.sqrt(2d / (outChannels + inChannels)),
        List(inChannels, outChannels, kernelSize, kernelSize),
        tOpt
      )
    )
    val biasVar = {
      val t = STen.zeros(List(outChannels), tOpt)
      if (bias) param(t) else const(t)
    }
    Conv2DTransposed(
      weightVar,
      biasVar,
      stride,
      padding,
      dilation
    )

  }

}
