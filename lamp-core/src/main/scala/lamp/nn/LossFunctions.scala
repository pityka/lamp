package lamp.nn

import lamp.autograd.Variable
import aten.Tensor
import lamp.autograd.Reduction
import lamp.autograd.Mean
import aten.ATen
import lamp.autograd.Sum
import lamp.syntax

trait LossFunction {

  /**
    * Returns the loss averaged over examples and the number of examples
    */
  def apply(
      output: Variable,
      target: Tensor,
      reduction: Reduction
  ): (Variable, Long)

}

object LossFunctions {

  case object MSE extends LossFunction {
    def apply(out: Variable, target: Tensor, reduction: Reduction) = {
      val v = out.mseLoss(target, reduction)
      (v, out.shape(0))
    }
  }
  case object L1Loss extends LossFunction {
    def apply(out: Variable, target: Tensor, reduction: Reduction) = {
      val v = out.l1Loss(target, reduction)
      (v, out.shape(0))
    }
  }
  case class NLL(
      numClasses: Int,
      classWeights: Tensor,
      ignore: Long = -100L
  ) extends LossFunction {
    def apply(out: Variable, target: Tensor, reduction: Reduction) = {
      val v = out.nllLoss(target, numClasses, classWeights, reduction, ignore)
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
      classWeights: Tensor,
      ignore: Long = -100L
  ) extends LossFunction {
    val ignoreScalar = Tensor.scalarLong(ignore, classWeights.options)
    def apply(out: Variable, target: Tensor, reduction: Reduction) = {
      assert(
        reduction == Mean,
        "Only Mean reduction is implemented for SequenceNLL"
      )
      val timeSteps = out.shape(0)
      val batches = out.shape(1)
      val lossesAtTimeSteps = (0 until timeSteps.toInt).map { t =>
        val t1 = ATen.select(target, 0, t)
        val ignored = ATen.eq_1(t1, ignoreScalar)
        val sumT = ATen.sum_0(ignored)
        val ignoredCount = sumT.toLongMat.raw(0)
        val count = batches - ignoredCount
        Tensor.releaseAll(Array(sumT, ignored))
        val v = out
          .select(0, t)
          .nllLoss(
            t1,
            numClasses,
            classWeights,
            reduction = Sum,
            ignore = ignore
          )
        (v.releaseWith(t1), count)
      }
      val totalCount = lossesAtTimeSteps.map(_._2).sum
      val v =
        lossesAtTimeSteps.map(_._1).reduce(_ + _) * (1d / totalCount)
      (v, totalCount)
    }
  }
}
