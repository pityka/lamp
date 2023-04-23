package lamp.nn

import aten.Tensor
import aten.ATen
import lamp.STen
import lamp.Scope
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
    clip0: Option[Double] = None
) extends Optimizer {
  val scope = Scope.free
    val clip = clip0.map(theta => STen.scalarDouble(theta,parameters.head._1.options(scope))(scope))

  val velocity: Seq[Option[(STen, OptimizerHyperparameter)]] =
    parameters.toList.map { case (param, _) =>
      momentum.map { m =>
        (STen.owned(Tensor.zeros_like(param.value))(scope), m)
      }
    }

  def state = velocity.flatMap(_.toList).map(_._1)

  def load(tensors: Seq[STen]) = {
    state.zip(tensors).foreach { case (current, incoming) =>
      current.copyFrom(incoming)
    }
  }
  def release() = {
    scope.release()
  }
  def step(gradients: Seq[Option[STen]], scheduleFactor: Double) = {
    clip.foreach { theta => gradientClippingInPlace(gradients, theta) }

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

        velocity.value.mul_(m)

        ATen.add_out(
          velocity.value,
          velocity.value,
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
          velocity.value,
          -1
        )
      case _ => ???
    }

  }
}
