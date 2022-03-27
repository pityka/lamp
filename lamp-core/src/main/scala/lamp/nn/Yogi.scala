package lamp.nn

import aten.Tensor
import aten.ATen
import lamp.STen
import lamp.Scope
import lamp.STenOptions

object Yogi {
  def factory(
      weightDecay: OptimizerHyperparameter,
      learningRate: OptimizerHyperparameter = simple(0.01),
      beta1: OptimizerHyperparameter = simple(0.9),
      beta2: OptimizerHyperparameter = simple(0.999),
      eps: Double = 1e-3,
      clip: Option[Double] = None,
      debias: Boolean = true
  ) =
    (parameters: Seq[(STen, PTag)]) =>
      Yogi(
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

/** The Yogi optimizer algorithm I added the decoupled weight decay term
  * following https://arxiv.org/pdf/1711.05101.pdf
  * @see
  *   https://papers.nips.cc/paper/2018/file/90365351ccc7437a1309dc64e4db32a3-Paper.pdf
  *   Algorithm 2
  */
case class Yogi(
    parameters: Seq[(STen, PTag)],
    weightDecay: OptimizerHyperparameter,
    learningRate: OptimizerHyperparameter = simple(0.01),
    beta1: OptimizerHyperparameter = simple(0.9),
    beta2: OptimizerHyperparameter = simple(0.999),
    eps: Double = 1e-3,
    clip: Option[Double] = None,
    debias: Boolean = true
) extends Optimizer {
  val scope = Scope.free
  val mt: List[STen] = parameters.toList.map { case (param, _) =>
    STen.owned(Tensor.zeros_like(param.value))(scope)
  }
  val vt: List[STen] = parameters.toList.map { case (param, _) =>
    STen.owned(Tensor.zeros_like(param.value))(scope)
  }

  var stepCount = 0L
  val stepCountSTen = STen.scalarDouble(0, STenOptions.d)(scope)

  def state = List(stepCountSTen) ++ mt ++ vt

  def load(tensors: Seq[STen]) = {
    state.zip(tensors).foreach { case (current, incoming) =>
      current.copyFrom(incoming)
    }
    stepCount = stepCountSTen.toDoubleArray.apply(0).toLong

  }
  def release() = {
    scope.release()
  }
  def step(gradients: Seq[Option[STen]], scheduleFactor: Double) = {
    clip.foreach { theta => gradientClippingInPlace(gradients, theta) }
    stepCount += 1
    stepCountSTen += 1d
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

          Scope.root { implicit scope =>
            mt.value.mul_(b1)
            ATen.add_out(mt.value, mt.value, gradients.value, (1d - b1))
            gradients.pow_(2d)
            val tmp = STen.owned(ATen.sub_0(vt.value, gradients.value, 1d))
            tmp.sign_()
            gradients *= tmp
            ATen.sub_out(vt.value, vt.value, gradients.value, 1d - b2)

            val debiasTerm =
              if (debias)
                1d / (1 - math.pow(b1, stepCount.toDouble))
              else 1d

            val stepParam =
              scheduleFactor * lr * debiasTerm

            val stepWd = scheduleFactor * wd

            val denom = STen.owned(ATen.sqrt(vt.value))
            if (debias) {
              denom *= 1d / math.sqrt(
                (1 - math.pow(b2, stepCount.toDouble))
              )
            }
            denom += eps

            if (wd != 0d) {
              STen.addOut(param, param, param, -1 * stepWd)
            }

            ATen.addcdiv_out(
              param.value,
              param.value,
              mt.value,
              denom.value,
              -1 * stepParam
            )
          }

      }
  }
}
