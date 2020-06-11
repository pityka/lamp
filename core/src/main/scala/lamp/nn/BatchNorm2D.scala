package lamp.nn

import lamp.autograd.{Variable, BatchNorm2D => BN, param, const}
import aten.Tensor
import aten.ATen
import aten.TensorOptions

case class BatchNorm2D(
    weight: Variable,
    bias: Variable,
    runningMean: Variable,
    runningVar: Variable,
    training: Boolean,
    momentum: Double,
    eps: Double
) extends Module {
  override def asTraining = copy(training = true)
  override def asEval = copy(training = false)

  override def load(parameters: Seq[Tensor]) = {
    val w = param(parameters.head)
    val b = param(parameters(1))
    val rm = const(parameters(2))
    val rv = const(parameters(3))
    copy(weight = w, bias = b, runningMean = rm, runningVar = rv)
  }

  override val state = List(
    weight -> BatchNorm.Weights,
    bias -> BatchNorm.Bias,
    runningMean -> BatchNorm.RunningMean,
    runningVar -> BatchNorm.RunningVar
  )

  override def forward(x: Variable): Variable =
    BN(
      x,
      weight,
      bias,
      runningMean.value,
      runningVar.value,
      training,
      momentum,
      eps
    ).value

}

object BatchNorm2D {
  case object Weights extends LeafTag
  case object Bias extends LeafTag
  def apply(
      features: Int,
      tOpt: TensorOptions,
      training: Boolean = true,
      momentum: Double = 0.1,
      eps: Double = 1e-5
  ): BatchNorm2D = BatchNorm2D(
    weight = param(ATen.normal_3(0.0, 0.01, Array(features.toLong), tOpt)),
    bias = param(ATen.zeros(Array(features.toLong), tOpt)),
    runningMean = const(ATen.zeros(Array(features.toLong), tOpt)),
    runningVar = const(ATen.zeros(Array(features.toLong), tOpt)),
    training = training,
    momentum = momentum,
    eps = eps
  )
}
