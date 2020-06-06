package lamp.nn

import lamp.autograd.{Variable, param, Conv2D => Conv2dOp, const}
import aten.{ATen, TensorOptions}
import lamp.autograd.TensorHelpers

case class Conv2D(
    weights: Variable,
    bias: Variable,
    stride: Long,
    padding: Long,
    dilation: Long,
    groups: Long
) extends Module {

  val parameters = List(
    weights -> Conv2D.Weights,
    bias -> Conv2D.Bias
  )

  def forward(x: Variable): Variable =
    Conv2dOp(x, weights, bias, stride, padding, dilation, groups).value

}

object Conv2D {
  case object Weights extends LeafTag
  case object Bias extends LeafTag
  def apply(
      inChannels: Long,
      outChannels: Long,
      kernelSize: Long,
      bias: Boolean = true,
      stride: Long = 1,
      padding: Long = 0,
      dilation: Long = 1,
      groups: Long = 1,
      tOpt: TensorOptions = TensorOptions.dtypeDouble
  ): Conv2D = {
    val weightVar = param(
      ATen.rand(Array(outChannels, inChannels, kernelSize, kernelSize), tOpt)
    )
    val biasVar = {
      val t = ATen.rand(Array(outChannels), tOpt)
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
