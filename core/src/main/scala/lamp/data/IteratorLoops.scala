package lamp.data

import aten.Tensor
import lamp.autograd._
import aten.ATen
import aten.TensorOptions
import lamp.nn._

object IteratorLoops {

  def iteratorEpochs(
      model: SupervisedModel,
      optimizerFactory: Seq[(Tensor, PTag)] => Optimizer,
      trainBatchesOverEpoch: () => Iterator[(Tensor, Tensor)],
      validationBatchesOverEpoch: () => Iterator[(Tensor, Tensor)],
      epochs: Int
  )(callbackOnValidationOutputAndTarget: (Tensor, Tensor, Double) => Unit) = {
    var epoch = 0L
    val modelWithOptimizer = model.zipOptimizer(optimizerFactory)

    var currentValidation = validationBatchesOverEpoch()
    while (epoch < epochs) {
      var currentTrain = trainBatchesOverEpoch()
      iteratorOneEpoch(modelWithOptimizer, trainBatchesOverEpoch())(trainLoss =>
        () // println(s"Training loss $trainLoss")
      )

      if (currentValidation.hasNext) {
        val (validationSample, validationTarget) = currentValidation.next
        val (validationLoss, validationOutput) = model.lossAndOutput(
          validationSample,
          validationTarget
        )
        callbackOnValidationOutputAndTarget(
          validationOutput,
          validationTarget,
          validationLoss
        )
        validationSample.release
        validationTarget.release
        validationOutput.release

      } else {
        currentValidation = validationBatchesOverEpoch()
      }

      epoch += 1
    }

  }

  def iteratorOneEpoch(
      model: ModelWithOptimizer,
      trainBatches: Iterator[(Tensor, Tensor)]
  )(callback: Double => Unit) = {
    trainBatches.foreach {
      case (sample, target) =>
        val (loss, gradients) =
          model.model.lossAndGradients(sample, target)
        callback(loss)
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
