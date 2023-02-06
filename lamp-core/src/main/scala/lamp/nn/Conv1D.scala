package lamp.nn

import lamp.autograd.{Variable, Constant, param, Convolution, const}
import lamp.Sc
import lamp.STenOptions
import lamp.scope
import lamp.STen

case class Conv1D(
    weights: Constant,
    bias: Constant,
    stride: Long,
    padding: Long,
    dilation: Long,
    groups: Long
) extends Module {

  override val state = List(
    weights -> Conv1D.Weights,
    bias -> Conv1D.Bias
  )

  def forward[S: Sc](x: Variable): Variable =
    new Convolution(
      scope = scope,
      input = x,
      weight = weights,
      bias = bias,
      stride = Array(stride),
      padding = Array(padding),
      dilation = Array(dilation),
      transposed = false,
      outputPadding = Array(0),
      groups = groups
    ).value

}

object Conv1D {
  implicit val trainingMode : TrainingMode[Conv1D] = TrainingMode.identity[Conv1D]
  implicit val load : Load[Conv1D] = Load.make[Conv1D](m =>
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
      bias: Boolean = true,
      stride: Long = 1,
      padding: Long = 0,
      dilation: Long = 1,
      groups: Long = 1
  ): Conv1D = {
    val weightVar = param(
      STen.normal(
        0d,
        math.sqrt(2d / (outChannels + inChannels)),
        List(outChannels, inChannels, kernelSize),
        tOpt
      )
    )
    val biasVar = {
      val t = STen.zeros(List(outChannels), tOpt)
      if (bias) param(t) else const(t)
    }
    Conv1D(
      weightVar,
      biasVar,
      stride,
      padding,
      dilation,
      groups
    )

  }

}
