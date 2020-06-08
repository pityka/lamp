package lamp.nn

import lamp.autograd.{Variable, BatchNorm => BN, param}
import aten.Tensor
import aten.ATen
import aten.TensorOptions

case class BatchNorm(
    weight: Variable,
    bias: Variable,
    runningMean: Tensor,
    runningVar: Tensor,
    training: Boolean,
    momentum: Double,
    eps: Double
) extends Module {
  override def asTraining = copy(training = true)
  override def asEval = copy(training = false)

  override def load(parameters: Seq[Tensor]) = {
    val w = param(parameters.head)
    val b = param(parameters(1))
    copy(weight = w, bias = b)
  }

  override val parameters = List(
    weight -> BatchNorm.Weights,
    bias -> BatchNorm.Bias
  )

  override def forward(x: Variable): Variable =
    BN(x, weight, bias, runningMean, runningVar, training, momentum, eps).value

}

object BatchNorm {
  case object Weights extends LeafTag
  case object Bias extends LeafTag
  def apply(
      features: Int,
      training: Boolean = true,
      momentum: Double = 0.1,
      eps: Double = 1e-5,
      tOpt: TensorOptions = TensorOptions.dtypeDouble()
  ): BatchNorm = BatchNorm(
    weight = param(ATen.normal_3(0.0, 0.01, Array(features.toLong), tOpt)),
    bias = param(ATen.zeros(Array(features.toLong), tOpt)),
    runningMean = ATen.zeros(Array(features.toLong), tOpt),
    runningVar = ATen.zeros(Array(features.toLong), tOpt),
    training = training,
    momentum = momentum,
    eps = eps
  )
}
