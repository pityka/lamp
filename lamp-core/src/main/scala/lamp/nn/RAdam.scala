package lamp.nn

import aten.Tensor
import aten.ATen
import lamp.STen

object RAdam {
  def factory(
      weightDecay: OptimizerHyperparameter,
      learningRate: OptimizerHyperparameter = simple(0.001),
      beta1: OptimizerHyperparameter = simple(0.9),
      beta2: OptimizerHyperparameter = simple(0.999),
      eps: Double = 1e-8,
      clip: Option[Double] = None
  ) =
    (parameters: Seq[(STen, PTag)]) =>
      RAdam(
        parameters,
        weightDecay,
        learningRate,
        beta1,
        beta2,
        eps,
        clip
      )
}

/** Rectified Adam optimizer algorithm
  */
case class RAdam(
    parameters: Seq[(STen, PTag)],
    weightDecay: OptimizerHyperparameter,
    learningRate: OptimizerHyperparameter = simple(0.001),
    beta1: OptimizerHyperparameter = simple(0.9),
    beta2: OptimizerHyperparameter = simple(0.999),
    eps: Double = 1e-8,
    clip: Option[Double] = None
) extends Optimizer {
  val mt: List[Tensor] = parameters.toList.map { case (param, _) =>
    Tensor.zeros_like(param.value)
  }
  val vt: List[Tensor] = parameters.toList.map { case (param, _) =>
    Tensor.zeros_like(param.value)
  }

  var stepCount = 0L
  def release() = {
    mt.foreach(_.release())
    vt.foreach(_.release())
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

          mt.mul_(b1)
          ATen.add_out(mt, mt, gradients.value, (1d - b1))

          vt.mul_(b2)
          ATen.addcmul_out(vt, vt, gradients.value, gradients.value, 1 - b2)

          val beta2PowT = math.pow(b2, stepCount.toDouble)

          val rhoInf = (2d / (1d - b2)) - 1
          val rho = rhoInf - (2d * stepCount * beta2PowT / (1d - beta2PowT))
          // println("--")
          // println("b" + beta2PowT)
          // println("r-i" + rhoInf)
          // println("r" + rho)

          val stepWd = scheduleFactor * wd
          if (wd != 0d) {
            ATen.add_out(param.value, param.value, param.value, -1 * stepWd)
          }

          if (rho > 4) {
            val stepParam = scheduleFactor * lr * math.sqrt(
              (1 - beta2PowT) * ((rho - 4) / (rhoInf - 4)) * ((rho - 2) / rho) * (rhoInf / (rhoInf - 2))
            ) / (1 - math.pow(b1, stepCount.toDouble))

            // println("A")
            // println(stepParam)

            val denom = ATen.sqrt(vt)
            denom.add_(eps, 1d)

            ATen.addcdiv_out(
              param.value,
              param.value,
              mt,
              denom,
              -1 * stepParam
            )
            denom.release

          } else {
            // println("B")
            val stepParam =
              scheduleFactor * lr / (1 - math.pow(b1, stepCount.toDouble))
            ATen.add_out(param.value, param.value, mt, -1 * stepParam)
          }

      }
  }
}
