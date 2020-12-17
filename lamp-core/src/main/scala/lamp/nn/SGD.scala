package lamp.nn

import aten.Tensor
import aten.ATen
import lamp.STen
object SGDW {
  def factory(
      learningRate: OptimizerHyperparameter,
      weightDecay: OptimizerHyperparameter,
      momentum: Option[OptimizerHyperparameter] = None,
      clip: Option[Double] = None
  ) =
    (parameters: Seq[(STen, PTag)]) =>
      SGDW(parameters, learningRate, weightDecay, momentum, clip)
}

// https://arxiv.org/pdf/1711.05101.pdf algorithm 1
case class SGDW(
    parameters: Seq[(STen, PTag)],
    learningRate: OptimizerHyperparameter,
    weightDecay: OptimizerHyperparameter,
    momentum: Option[OptimizerHyperparameter] = None,
    clip: Option[Double] = None
) extends Optimizer {
  val velocity: Seq[Option[(Tensor, OptimizerHyperparameter)]] =
    parameters.toList.map {
      case (param, _) =>
        momentum.map { m => (Tensor.zeros_like(param.value), m) }
    }

  var stepCount = 0L
  def release = {
    velocity.foreach(_.foreach(_._1.release))
  }
  def step(gradients: Seq[Option[STen]], scheduleFactor: Double) = {
    clip.foreach { theta => gradientClippingInPlace(gradients, theta) }
    stepCount += 1L

    parameters.zip(gradients).zip(velocity).filter(_._1._2.isDefined).foreach {
      case (((param, tag), Some(gradients)), None) =>
        val wd = weightDecay(tag)
        if (wd != 0d) {
          ATen.add_out(
            param.value,
            param.value,
            param.value,
            (-1) * wd * scheduleFactor
          )
        }

        ATen.add_out(
          param.value,
          param.value,
          gradients.value,
          (-1) * learningRate(tag) * scheduleFactor
        )

      case (((param, tag), Some(gradients)), Some((velocity, momentum))) =>
        val m = momentum(tag)

        velocity.mul_(m)

        ATen.add_out(
          velocity,
          velocity,
          gradients.value,
          learningRate(tag) * scheduleFactor
        )

        val wd = weightDecay(tag)
        if (wd != 0d) {
          ATen.add_out(
            param.value,
            param.value,
            param.value,
            -1 * wd * scheduleFactor
          )
        }
        ATen.add_out(
          param.value,
          param.value,
          velocity,
          -1
        )
      case _ => ???
    }

  }
}
