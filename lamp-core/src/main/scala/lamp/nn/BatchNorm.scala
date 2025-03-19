package lamp.nn

import lamp.autograd.{Variable, Constant, BatchNorm => BN, param, const}
import lamp.Sc
import lamp.scope
import lamp.STen
import lamp.STenOptions

case class BatchNorm(
    weight: Constant,
    bias: Constant,
    runningMean: Constant,
    runningVar: Constant,
    training: Boolean,
    momentum: Double,
    eps: Double,
    forceTrain: Boolean,
    forceEval: Boolean,
    evalIfBatchSizeIsOne: Boolean
) extends Module {

  override val state = List(
    weight -> BatchNorm.Weights,
    bias -> BatchNorm.Bias,
    runningMean -> BatchNorm.RunningMean,
    runningVar -> BatchNorm.RunningVar
  )

  override def forward[S: Sc](x: Variable): Variable = {
    val tr =
      if (forceTrain) true
      else if (forceEval) false
      else if (evalIfBatchSizeIsOne && x.shape(0) == 1) false
      else training
    new BN(
      scope,
      x,
      weight,
      bias,
      runningMean.value,
      runningVar.value,
      tr,
      momentum,
      eps
    ).value
  }

}

object BatchNorm {
  implicit val trainingMode: TrainingMode[BatchNorm] =
    TrainingMode.make[BatchNorm](
      asEval1 = m => m.copy(training = false),
      asTraining1 = m => m.copy(training = true)
    )
  implicit val load: Load[BatchNorm] = Load.make[BatchNorm](m =>
    tensors => {
      m.weight.value.copyFrom(tensors.head)
      m.bias.value.copyFrom(tensors(1))
      m.runningMean.value.copyFrom(tensors(2))
      m.runningVar.value.copyFrom(tensors(3))

    }
  )
  case object Weights extends LeafTag
  case object Bias extends LeafTag
  case object RunningMean extends LeafTag
  case object RunningVar extends LeafTag
  def apply[S: Sc](
      features: Int,
      tOpt: STenOptions,
      training: Boolean = true,
      momentum: Double = 0.1,
      eps: Double = 1e-5,
      forceTrain: Boolean = false,
      forceEval: Boolean = false,
      evalIfBatchSizeIsOne: Boolean = false
  ): BatchNorm = BatchNorm(
    weight = param(STen.normal(0.0, 0.01, List(features.toLong), tOpt)),
    bias = param(STen.zeros(List(features.toLong), tOpt)),
    runningMean = const(STen.zeros(List(features.toLong), tOpt)),
    runningVar = const(STen.zeros(List(features.toLong), tOpt)),
    training = training,
    momentum = momentum,
    eps = eps,
    forceTrain = forceTrain,
    forceEval = forceEval,
    evalIfBatchSizeIsOne = evalIfBatchSizeIsOne
  )
}
