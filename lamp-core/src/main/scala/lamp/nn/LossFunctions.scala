package lamp.nn

import lamp.autograd.Variable
import aten.Tensor
import lamp.autograd.Reduction
import lamp.autograd.Mean
import aten.ATen
import lamp.autograd.Sum
import lamp.util.syntax
import lamp.Sc
import lamp.STen

trait LossFunction {

  /** Returns the loss averaged over examples and the number of examples
    */
  def apply[S: Sc](output: Variable, target: STen): (Variable, Long)
}

object LossFunctions {
  /* Ignores target (except shape(0)), useful if actual loss is computed upstream */
  case object Identity extends LossFunction {
    def apply[S: Sc](output: Variable, target: STen): (Variable, Long) =
      (output, target.shape(0))
  }
  case object MSE extends LossFunction {
    def apply[S: Sc](out: Variable, target: STen) = {
      val v = out.mseLoss(target)
      (v, out.shape(0))
    }
  }
  case class SmoothL1Loss(reduction: Reduction = Mean, beta: Double = 1.0)
      extends LossFunction {
    def apply[S: Sc](out: Variable, target: STen) = {
      val v = out.smoothL1Loss(target, reduction, beta)
      (v, out.shape(0))
    }
  }
  case class NLL(
      numClasses: Int,
      classWeights: STen,
      reduction: Reduction = Mean,
      ignore: Long = -100L
  ) extends LossFunction {
    def apply[S: Sc](out: Variable, target: STen) = {
      val v = out.nllLoss(
        target,
        // numClasses,
        classWeights,
        reduction,
        ignore
      )
      (v, out.shape(0))
    }
  }

  case class BCEWithLogits(
      posWeights: Option[STen] = None,
      reduction: Reduction = Mean,
      ignore: Long = -100L
  ) extends LossFunction {
    def apply[S: Sc](out: Variable, target: STen) = {
      val v = out.binaryCrossEntropyWithLogitsLoss(
        target,
        posWeights,
        reduction
      )
      (v, out.shape(0))
    }
  }

  /** Return a loss function which takes outputs of time step x batch x classes
    * and targets of time step x batch The returned loss is averaged over the
    * batch and the time steps
    */
  case class SequenceNLL(
      numClasses: Int,
      classWeights: STen,
      ignore: Long = -100L
  ) extends LossFunction {
    def apply[S: Sc](out: Variable, target: STen) = {
      val ignoreScalar = Tensor.scalarLong(ignore, classWeights.options.value)
      val timeSteps = out.shape(0)
      val batches = out.shape(1)
      val lossesAtTimeSteps = (0 until timeSteps.toInt).map { t =>
        val t1 = STen.owned(ATen.select(target.value, 0, t))
        val ignored = ATen.eq_1(t1.value, ignoreScalar)
        val sumT = ATen.sum_0(ignored)
        val ignoredCount = sumT.toLongArray.apply(0)
        val count = batches - ignoredCount
        Tensor.releaseAll(Array(sumT, ignored))
        val v = out
          .select(0, t)
          .nllLoss(
            t1,
            // numClasses,
            classWeights,
            reduction = Sum,
            ignore = ignore
          )
        (v, count)
      }
      val totalCount = lossesAtTimeSteps.map(_._2).sum
      val v =
        lossesAtTimeSteps.map(_._1).reduce(_ + _) * (1d / totalCount)
      (v, totalCount)
    }
  }
}
