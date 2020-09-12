package lamp.nn

import lamp.autograd._
import aten.ATen
import aten.Tensor
import lamp.syntax

/**
  * Input gradient regularization with finite differences
  * https://arxiv.org/pdf/1905.11468.pdf
  */
case object InputGradientRegularization {
  def outputAndLoss[M <: Module](
      module: M with Module,
      lossFunction: LossFunction
  )(
      input: Variable,
      target: Tensor,
      h: Double,
      lambda: Double
  ) = {
    val inputWithGrad = input.copy(needsGrad = true)
    val output = module.forward(inputWithGrad)
    val (loss1, examples) = lossFunction(output, target, NoReduction)
    val loss1Mean = loss1.mean(List(0))
    val z = {
      val inputGradient = {
        inputWithGrad.zeroGrad()
        module.parameters.foreach {
          case (param, _) =>
            param.zeroGrad()
        }

        loss1Mean.backprop()
        val g = inputWithGrad.partialDerivative
        ATen.clone(g.get)
      }

      val norm = ATen.norm_3(inputGradient, Array(0L), true)
      val normRec = ATen.reciprocal(norm)
      val normZero = ATen.eq_0(norm, 0.0)
      val zeros = ATen.zeros_like(normRec, normRec.options)
      val normOrZero = ATen.where_0(normZero, zeros, normRec)
      ATen.mul_out(inputGradient, inputGradient, normOrZero)
      norm.release
      normRec.release
      normOrZero.release
      normZero.release
      zeros.release
      val d = const(inputGradient)(input.pool).needsNoGrad.releasable
      input + d * h

    }

    val lossAtZ = {
      val outputAtZ = module.forward(z)
      val (loss, _) = lossFunction(outputAtZ, target, NoReduction)
      loss
    }

    val r = {
      ((lossAtZ - loss1).pow(2d) * (1 / (h * h))).mean(List(0))
    }
    // println((loss1Mean.toMat.raw(0), r.toMat.raw(0)))
    val regularizedLoss = loss1Mean + r * lambda
    (output, regularizedLoss, examples)
  }
}
