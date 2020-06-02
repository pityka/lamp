package lamp.data

import aten.Tensor
import lamp.autograd._
import aten.ATen
import aten.TensorOptions
import lamp.nn._

trait TrainingCallback {
  def apply(trainingLoss: Double, batchCount: Int): Unit
}
object TrainingCallback {
  val noop = new TrainingCallback {
    def apply(trainingLoss: Double, batchCount: Int) = ()
  }
}

trait ValidationCallback {
  def apply(
      validationOutput: Tensor,
      validationTarget: Tensor,
      validationLoss: Double,
      epochCount: Long
  ): Unit
}
object ValidationCallback {
  val noop = new ValidationCallback {
    def apply(
        validationOutput: Tensor,
        validationTarget: Tensor,
        validationLoss: Double,
        epochCount: Long
    ) = ()
  }
}

object IteratorLoops {

  def iteratorEpochs(
      model: SupervisedModel,
      optimizerFactory: Seq[(Tensor, PTag)] => Optimizer,
      trainBatchesOverEpoch: () => Iterator[(Tensor, Tensor)],
      validationBatchesOverEpoch: () => Iterator[(Tensor, Tensor)],
      epochs: Int,
      trainingCallback: TrainingCallback,
      validationCallbackk: ValidationCallback
  ) = {
    var epoch = 0L
    val modelWithOptimizer = model.zipOptimizer(optimizerFactory)

    var currentValidation = validationBatchesOverEpoch()
    while (epoch < epochs) {
      var currentTrain = trainBatchesOverEpoch()
      iteratorOneEpoch(
        modelWithOptimizer,
        trainBatchesOverEpoch(),
        trainingCallback
      )

      if (currentValidation.hasNext) {
        val (validationSample, validationTarget) = currentValidation.next
        val (validationLoss, validationOutput) = model.lossAndOutput(
          validationSample,
          validationTarget
        )
        validationCallbackk(
          validationOutput,
          validationTarget,
          validationLoss,
          epoch
        )
        validationSample.release
        validationTarget.release
        validationOutput.release

      } else {
        currentValidation = validationBatchesOverEpoch()
      }

      epoch += 1
    }

    modelWithOptimizer.model

  }

  def iteratorOneEpoch(
      model: ModelWithOptimizer,
      trainBatches: Iterator[(Tensor, Tensor)],
      trainingCallback: TrainingCallback
  ) = {
    trainBatches.zipWithIndex.foreach {
      case ((sample, target), idx) =>
        val (loss, gradients) =
          model.model.lossAndGradients(sample, target)
        trainingCallback(loss, idx)
        model.optimizer.step(gradients)
        sample.release()
        target.release()
    }
  }
}

sealed trait Device {
  def to(t: Tensor): Tensor
  def options: TensorOptions
}
case object CPU extends Device {
  def to(t: Tensor) = {
    t.cpu
  }
  def options: TensorOptions = TensorOptions.d.cpu
}
case class CudaDevice(i: Int) extends Device {
  assert(
    i == 0,
    "Multi gpu not implemented. Implement Tensor.to(TensorOptions)."
  )
  def to(t: Tensor): Tensor = t.cuda
  def options: TensorOptions = TensorOptions.d.cuda_index(i)
}
