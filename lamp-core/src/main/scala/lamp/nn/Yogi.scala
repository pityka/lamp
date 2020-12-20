package lamp.nn

import aten.Tensor
import aten.ATen
import lamp.STen
import lamp.Scope

object Yogi {
  def factory(
      weightDecay: OptimizerHyperparameter,
      learningRate: OptimizerHyperparameter = simple(0.01),
      beta1: OptimizerHyperparameter = simple(0.9),
      beta2: OptimizerHyperparameter = simple(0.999),
      eps: Double = 1e-3,
      clip: Option[Double] = None
  ) =
    (parameters: Seq[(STen, PTag)]) =>
      Yogi(
        parameters,
        weightDecay,
        learningRate,
        beta1,
        beta2,
        eps,
        clip
      )
}

// https://papers.nips.cc/paper/2018/file/90365351ccc7437a1309dc64e4db32a3-Paper.pdf Algorithm 2
// I added the decoupled weight decay term following https://arxiv.org/pdf/1711.05101.pdf
case class Yogi(
    parameters: Seq[(STen, PTag)],
    weightDecay: OptimizerHyperparameter,
    learningRate: OptimizerHyperparameter = simple(0.01),
    beta1: OptimizerHyperparameter = simple(0.9),
    beta2: OptimizerHyperparameter = simple(0.999),
    eps: Double = 1e-3,
    clip: Option[Double] = None
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
        case ((((param, tag), Some(gradients)), mt), vt) =>
          val wd = weightDecay(tag)
          val b1 = beta1(tag)
          val b2 = beta2(tag)
          val lr = learningRate(tag)

          Scope.root { implicit scope =>
            mt.mul_(b1)
            ATen.add_out(mt, mt, gradients.value, (1d - b1))
            gradients.pow_(2d)
            val tmp = STen.owned(ATen.sub_0(vt, gradients.value, 1d))
            tmp.sign_()
            gradients *= tmp
            ATen.sub_out(vt, vt, gradients.value, 1d - b2)

            val mtcap =
              STen.owned(ATen.div_1(mt, (1 - math.pow(b1, stepCount))))
            val vtcap =
              STen.owned(ATen.div_1(vt, (1 - math.pow(b2, stepCount))))

            vtcap.sqrt_
            vtcap += eps
            mtcap /= vtcap
            mtcap *= lr
            if (wd != 0d) {
              STen.addOut(mtcap, mtcap, param, wd)
            }
            STen.addOut(param, param, mtcap, -1 * scheduleFactor)
          }

      }
  }
}
