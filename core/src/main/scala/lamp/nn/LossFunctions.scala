package lamp.nn

import lamp.autograd.Variable
import aten.Tensor
import lamp.autograd.Reduction
import lamp.autograd.Mean

object LossFunctions {
  def NLL(
      numClasses: Int,
      classWeights: Tensor,
      reduction: Reduction = Mean
  ) = { (out: Variable, target: Tensor) =>
    out.nllLoss(target, numClasses, classWeights, reduction)
  }
}
