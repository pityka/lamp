package lamp.nn

import lamp.STen
import lamp.Scope
import lamp.STenOptions

object AdamW {
  def factory(
      weightDecay: OptimizerHyperparameter,
      learningRate: OptimizerHyperparameter = simple(0.001),
      beta1: OptimizerHyperparameter = simple(0.9),
      beta2: OptimizerHyperparameter = simple(0.95),
      eps: Double = 1e-8,
      clip: Option[Double] = None,
      debias: Boolean = true,
      mixedPrecision: Boolean = false
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
        debias,
        mixedPrecision
      )
}

/** @see
  *   https://arxiv.org/pdf/1711.05101.pdf Algorithm 2
  */
case class AdamW(
    parameters: Seq[(STen, PTag)],
    weightDecay: OptimizerHyperparameter,
    learningRate: OptimizerHyperparameter = simple(0.001),
    beta1: OptimizerHyperparameter = simple(0.9),
    beta2: OptimizerHyperparameter = simple(0.999),
    eps: Double = 1e-8,
    clip0: Option[Double] = None,
    debias: Boolean = true,
    mixedPrecision: Boolean = false
) extends Optimizer {
  val scope0 = Scope.free

  def upCast(t: STen)(implicit scope: Scope) = {
    if (!mixedPrecision) t
    else
      t.scalarTypeByte match {
        case 5 | 15 => t.castToFloat
        case _      => t
      }
  }
  def downCast(t: STen, target: STen)(implicit scope: Scope) = {

    target.scalarTypeByte match {
      case 5  => t.castToType(5)
      case 15 => t.castToType(15)
      case _  => t
    }
  }

  val clip = clip0.map(theta =>
    STen.scalarDouble(theta, parameters.head._1.options(scope0))(scope0)
  )
  val workingCopy: List[Option[STen]] = parameters.toList.map {
    case (param, _) =>
      val copy = upCast(param)(scope0)
      if (copy.eq(param)) None else Some(copy)
  }
  val mt: List[STen] = parameters.toList.map { case (param, _) =>
    implicit val scope = scope0
    Scope { implicit scope =>
      upCast(STen.zerosLike(param))
    }

  }
  val vt: List[STen] = parameters.toList.map { case (param, _) =>
    implicit val scope = scope0
    Scope { implicit scope =>
      upCast(STen.zerosLike(param))
    }
  }

  def load(tensors: Seq[STen]) = {
    state.zip(tensors).foreach { case (current, incoming) =>
      current.copyFrom(incoming)
    }
    stepCount = stepCountSTen.toDoubleArray.apply(0).toLong

  }

  var stepCount = 0L
  val stepCountSTen = STen.scalarDouble(0, STenOptions.d)(scope0)
  def state = List(stepCountSTen) ++ mt ++ vt ++ workingCopy.flatMap(_.toList)
  def release() = {
    scope0.release()
  }
  def step(gradients: Seq[Option[STen]], scheduleFactor: Double) = {
    clip.foreach { theta => gradientClippingInPlace(gradients, theta) }
    stepCount += 1
    stepCountSTen += 1d

    parameters
      .zip(gradients)
      .zip(mt)
      .zip(vt)
      .zip(workingCopy)
      .filter(_._1._1._1._2.isDefined)
      .foreach {

        case (
              ((((paramInModel, tag), Some(gradients0)), mt), vt),
              paramWorkingCopy
            ) =>
          val wd = weightDecay(tag)
          val b1 = beta1(tag)
          val b2 = beta2(tag)
          val lr = learningRate(tag)

          Scope.root { implicit scope =>
            val gradients = upCast(gradients0)
            // L7
            mt *= b1

            STen.addOut(mt, mt, gradients, (1d - b1))

            // L8
            vt *= b2
            STen.addcmulOut(
              vt,
              vt,
              gradients,
              gradients,
              1 - b2
            )

            // L9-L12..
            val denom = vt.sqrt
            denom += eps

            val stepParam =
              if (debias)
                scheduleFactor * lr * math.sqrt(
                  (1 - math.pow(b2, stepCount.toDouble))
                ) / (1 - math.pow(b1, stepCount.toDouble))
              else scheduleFactor * lr

            val stepWd = scheduleFactor * wd

            val param = paramWorkingCopy.getOrElse(paramInModel)

            if (wd != 0d) {
              STen.addOut(param, param, param, -1 * stepWd)
            }

            STen.addcdivOut(
              param,
              param,
              mt,
              denom,
              -1 * stepParam
            )

            if (!param.eq(paramInModel)) {
              paramInModel.copyFrom(downCast(param, paramInModel))
            }

          }
        case _ =>
          // won't happen see filter above, suppressing warning
          ???
      }
  }
}
