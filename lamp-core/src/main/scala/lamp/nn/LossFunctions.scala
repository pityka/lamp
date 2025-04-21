package lamp.nn

import lamp.autograd.Variable

import lamp.autograd.Reduction
import lamp.autograd.Mean

import lamp.Sc
import lamp.STen

trait LossFunction {

  /** Returns the loss averaged over examples and the number of examples
    */
  def apply[S: Sc](output: Variable, target: STen): Variable
}

object LossFunctions {
  /* Ignores target (except shape(0)), useful if actual loss is computed upstream */
  case object Identity extends LossFunction {
    def apply[S: Sc](output: Variable, target: STen) =
      output
  }
  case object MSE extends LossFunction {
    def apply[S: Sc](out: Variable, target: STen) = {
      val v = out.mseLoss(target)
      v
    }
  }
  case class SmoothL1Loss(reduction: Reduction = Mean, beta: Double = 1.0)
      extends LossFunction {
    def apply[S: Sc](out: Variable, target: STen) = {
      val v = out.smoothL1Loss(target, reduction, beta)
      v
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
      v
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
      v
    }
  }

  
}
