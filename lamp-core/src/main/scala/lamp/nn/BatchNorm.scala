package lamp.nn

import lamp.autograd.{
  Variable,
  BatchNorm => BN,
  param,
  const,
  AllocatedVariablePool
}
import aten.ATen
import aten.TensorOptions

case class BatchNorm(
    weight: Variable,
    bias: Variable,
    runningMean: Variable,
    runningVar: Variable,
    training: Boolean,
    momentum: Double,
    eps: Double
) extends Module {

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

object BatchNorm {
  implicit val trainingMode = TrainingMode.make[BatchNorm](
    asEval1 = m => m.copy(training = false),
    asTraining1 = m => m.copy(training = true)
  )
  implicit val load = Load.make[BatchNorm](m =>
    tensors => {
      implicit val pool = m.weight.pool
      val w = param(tensors.head)
      val b = param(tensors(1))
      val rm = const(tensors(2))
      val rv = const(tensors(3))
      m.copy(weight = w, bias = b, runningMean = rm, runningVar = rv)
    }
  )
  case object Weights extends LeafTag
  case object Bias extends LeafTag
  case object RunningMean extends LeafTag
  case object RunningVar extends LeafTag
  def apply(
      features: Int,
      tOpt: TensorOptions,
      training: Boolean = true,
      momentum: Double = 0.1,
      eps: Double = 1e-5
  )(implicit pool: AllocatedVariablePool): BatchNorm = BatchNorm(
    weight = param(ATen.normal_3(0.0, 0.01, Array(features.toLong), tOpt)),
    bias = param(ATen.zeros(Array(features.toLong), tOpt)),
    runningMean = const(ATen.zeros(Array(features.toLong), tOpt)),
    runningVar = const(ATen.zeros(Array(features.toLong), tOpt)),
    training = training,
    momentum = momentum,
    eps = eps
  )
}
