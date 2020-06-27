package lamp.nn

import lamp.autograd.{Variable, param, Conv2D => Conv2dOp, const}
import aten.{ATen, TensorOptions}
import lamp.autograd.TensorHelpers
import aten.Tensor

case class Conv2D(
    weights: Variable,
    bias: Variable,
    stride: Long,
    padding: Long,
    dilation: Long,
    groups: Long
) extends Module {

  override val state = List(
    weights -> Conv2D.Weights,
    bias -> Conv2D.Bias
  )

  def forward(x: Variable): Variable =
    Conv2dOp(x, weights, bias, stride, padding, dilation, groups).value

}

object Conv2D {
  implicit val trainingMode = TrainingMode.identity[Conv2D]
  implicit val load = Load.make[Conv2D](m =>
    parameters => {
      val w = param(parameters.head)
      val b =
        if (m.bias.needsGrad) param(parameters(1)) else const(parameters(1))
      m.copy(weights = w, bias = b)
    }
  )
  case object Weights extends LeafTag
  case object Bias extends LeafTag
  def apply(
      inChannels: Long,
      outChannels: Long,
      kernelSize: Long,
      tOpt: TensorOptions,
      bias: Boolean = false,
      stride: Long = 1,
      padding: Long = 0,
      dilation: Long = 1,
      groups: Long = 1
  ): Conv2D = {
    val weightVar = param(
      ATen.normal_3(
        0d,
        math.sqrt(2d / (outChannels + inChannels)),
        Array(outChannels, inChannels, kernelSize, kernelSize),
        tOpt
      )
    )
    val biasVar = {
      val t = ATen.zeros(Array(outChannels), tOpt)
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
