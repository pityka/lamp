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
      scheduler: Long => Double = _ => 1d,
      clip: Option[Double] = None
  ) =
    (parameters: Seq[(STen, PTag)]) =>
      AdamW(
        parameters,
        weightDecay,
        learningRate,
        beta1,
        beta2,
        eps,
        scheduler,
        clip
      )
}

// https://arxiv.org/pdf/1711.05101.pdf Algorithm 2
case class AdamW(
    parameters: Seq[(STen, PTag)],
    weightDecay: OptimizerHyperparameter,
    learningRate: OptimizerHyperparameter = simple(0.001),
    beta1: OptimizerHyperparameter = simple(0.9),
    beta2: OptimizerHyperparameter = simple(0.999),
    eps: Double = 1e-8,
    scheduler: Long => Double = _ => 1d,
    clip: Option[Double] = None
) extends Optimizer {
  val mt: List[Tensor] = parameters.toList.map {
    case (param, _) => ATen.zeros_like(param.value, param.options)
  }
  val vt: List[Tensor] = parameters.toList.map {
    case (param, _) => ATen.zeros_like(param.value, param.options)
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
          ATen.pow_out_0(gradients.value, gradients.value, 2d)
          ATen.add_out(vt, vt, gradients.value, (1d - b2))

          // L9
          val mtcap = ATen.div_1(mt, 1d / (1 - math.pow(b1, stepCount)))

          // L10
          val vtcap = ATen.div_1(vt, 1d / (1 - math.pow(b2, stepCount)))

          // L11

          // L12
          ATen.sqrt_(vtcap)
          vtcap.add_(eps, 1d)
          ATen.div_out(mtcap, mtcap, vtcap)
          mtcap.mul_(lr)
          ATen.add_out(mtcap, mtcap, param.value, wd)
          ATen.add_out(param.value, param.value, mtcap, -1 * scheduleFactor)
          mtcap.release
          vtcap.release
      }
  }
}
