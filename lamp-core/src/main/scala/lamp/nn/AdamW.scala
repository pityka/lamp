package lamp.nn

import aten.Tensor
import aten.ATen
import lamp.STen

object AdamW {
  def factory(
      weightDecay: OptimizerHyperparameter,
      learningRate: OptimizerHyperparameter = simple(0.001),
      beta1: OptimizerHyperparameter = simple(0.9),
      beta2: OptimizerHyperparameter = simple(0.999),
      eps: Double = 1e-8,
      clip: Option[Double] = None,
      debias: Boolean = true
  ) =
    (parameters: Seq[(STen, PTag)]) =>
      AdamW(
        parameters,
        weightDecay,
        learningRate,
        beta1,
        beta2,
        eps,
        clip,
        debias
      )
}

/**
  * @see https://arxiv.org/pdf/1711.05101.pdf Algorithm 2
  */
case class AdamW(
    parameters: Seq[(STen, PTag)],
    weightDecay: OptimizerHyperparameter,
    learningRate: OptimizerHyperparameter = simple(0.001),
    beta1: OptimizerHyperparameter = simple(0.9),
    beta2: OptimizerHyperparameter = simple(0.999),
    eps: Double = 1e-8,
    clip: Option[Double] = None,
    debias: Boolean = true
) extends Optimizer {
  val mt: List[Tensor] = parameters.toList.map {
    case (param, _) => Tensor.zeros_like(param.value)
  }
  val vt: List[Tensor] = parameters.toList.map {
    case (param, _) => Tensor.zeros_like(param.value)
  }

  var stepCount = 0L
  def release = {
    mt.foreach(_.release)
    vt.foreach(_.release)
  }
  def step(gradients: Seq[Option[STen]], scheduleFactor: Double) = {
    clip.foreach { theta => gradientClippingInPlace(gradients, theta) }
    stepCount += 1
    parameters
      .zip(gradients)
      .zip(mt)
      .zip(vt)
      .filter(_._1._1._2.isDefined)
      .foreach {
        case ((((_, _), None), _), _) =>
          // won't happent, see filter above
          ???
        case ((((param, tag), Some(gradients)), mt), vt) =>
          val wd = weightDecay(tag)
          val b1 = beta1(tag)
          val b2 = beta2(tag)
          val lr = learningRate(tag)

          // L7
          mt.mul_(b1)
          ATen.add_out(mt, mt, gradients.value, (1d - b1))

          // L8
          vt.mul_(b2)
          ATen.addcmul_out(vt, vt, gradients.value, gradients.value, 1 - b2)

          // L9-L12..
          val denom = ATen.sqrt(vt)
          denom.add_(eps, 1d)

          val stepParam =
            if (debias)
              scheduleFactor * lr * math.sqrt(
                (1 - math.pow(b2, stepCount))
              ) / (1 - math.pow(b1, stepCount))
            else scheduleFactor * lr

          val stepWd = scheduleFactor * wd

          if (wd != 0d) {
            ATen.add_out(param.value, param.value, param.value, -1 * stepWd)
          }

          ATen.addcdiv_out(param.value, param.value, mt, denom, -1 * stepParam)

          denom.release
      }
  }
}
