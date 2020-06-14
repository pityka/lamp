package lamp.nn

import lamp.autograd.Variable
import aten.Tensor
import lamp.autograd.Reduction
import lamp.autograd.Mean
import aten.ATen
import lamp.autograd.Sum

trait LossFunction {

  /**
    * Returns the loss averaged over examples and the number of examples
    */
  def apply(output: Variable, target: Tensor): (Variable, Long)
}

object LossFunctions {
  case class NLL(
      numClasses: Int,
      classWeights: Tensor,
      reduction: Reduction = Mean
  ) extends LossFunction {
    def apply(out: Variable, target: Tensor) = {
      val v = out.nllLoss(target, numClasses, classWeights, reduction)
      (v, out.shape(0))
    }
  }

  /**
    * Return a loss function which takes outputs of time step x batch x classes
    * and targets of time step x batch
    * The returned loss is averaged over the batch and the time steps
    */
  case class SequenceNLL(
      numClasses: Int,
      classWeights: Tensor
  ) extends LossFunction {
    def apply(out: Variable, target: Tensor) = {
      val timeSteps = out.shape(0)
      val batches = out.shape(1)
      val lossesAtTimeSteps = (0 until timeSteps.toInt).map { t =>
        val t1 = ATen.select(target, 0, t)
        val v = out
          .select(0, t)
          .nllLoss(t1, numClasses, classWeights, reduction = Sum)
        v.releaseWith(t1)
      }
      val v = lossesAtTimeSteps.reduce(_ + _) * (1d / (timeSteps * batches))
      (v, timeSteps * batches)
    }
  }
}
