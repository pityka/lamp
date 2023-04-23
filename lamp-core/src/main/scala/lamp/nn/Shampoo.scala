package lamp.nn

import aten.ATen
import lamp.STen
import lamp.Scope
import lamp.STenOptions

object Shampoo {
  def factory(
      learningRate: OptimizerHyperparameter = simple(0.001),
      clip: Option[Double] = None,
      eps: Double = 1e-4,
      diagonalThreshold: Int = 256,
      updatePreconditionerEveryNIterations: Int = 100,
      momentum: OptimizerHyperparameter = simple(0d)
  ) =
    (parameters: Seq[(STen, PTag)]) =>
      Shampoo(
        parameters,
        learningRate,
        clip,
        eps,
        diagonalThreshold,
        updatePreconditionerEveryNIterations,
        momentum
      )
}

/** @see
  *   https://arxiv.org/pdf/1802.09568.pdf Algorithm 1
  */
case class Shampoo(
    parameters: Seq[(STen, PTag)],
    learningRate: OptimizerHyperparameter = simple(0.001),
    clip0: Option[Double] = None,
    eps: Double = 1e-4,
    diagonalThreshold: Int = 256,
    updatePreconditionerEveryNIterations: Int = 100,
    momentum: OptimizerHyperparameter = simple(0d)
) extends Optimizer {
  val scope = Scope.free
  val clip = clip0.map(theta => STen.scalarDouble(theta,parameters.head._1.options(scope))(scope))

  val lt: List[(STen, STen)] = parameters.toList.map { case (param, _) =>
    val dim1 = param.shape(0)
    val t =
      if (dim1 > 512) STen.ones(List(dim1.toInt), param.options(scope))(scope)
      else
        STen.eye(dim1.toInt, param.options(scope))(scope)
    t *= eps
    val ltinv = t.cloneTensor(scope)
    (t, ltinv)
  }
  val rt: List[(STen, STen)] = parameters.toList.map { case (param, _) =>
    val dim2 = param.shape.drop(1).foldLeft(1L)(_ * _)
    val t =
      if (dim2 > 512) STen.ones(List(dim2.toInt), param.options(scope))(scope)
      else
        STen.eye(dim2.toInt, param.options(scope))(scope)
    t *= eps
    (t, t.cloneTensor(scope))
  }
  val lastGradient: List[STen] = parameters.toList.map { case (param, _) =>
    val t =
      STen.zerosLike(param)(scope)
    t
  }

  def load(tensors: Seq[STen]) = {
    state.zip(tensors).foreach { case (current, incoming) =>
      current.copyFrom(incoming)
    }
    stepCount = stepCountSTen.toDoubleArray.apply(0).toLong

  }

  var stepCount = 0L
  val stepCountSTen = STen.scalarDouble(0, STenOptions.d)(scope)
  def state =
    List(stepCountSTen) ++ lt.flatMap(v => List(v._1, v._2)) ++ rt.flatMap(v =>
      List(v._1, v._2)
    ) ++ lastGradient
  def release() = {
    scope.release()
  }
  def step(gradients: Seq[Option[STen]], scheduleFactor: Double) = {
    clip.foreach { theta => gradientClippingInPlace(gradients, theta) }
    stepCount += 1
    stepCountSTen += 1d
    parameters
      .zip(gradients)
      .zip(lt)
      .zip(rt)
      .zip(lastGradient)
      .filter(_._1._1._1._2.isDefined)
      .foreach {
        case (((((_, _), None), _), _), _) =>
          // won't happent, see filter above
          ???
        case (
              ((((param, tag), Some(gradients)), (lt, ltinv)), (rt, rtinv)),
              lastGradient
            ) =>
          val lr = learningRate(tag)
          val mom = momentum(tag)

          if (stepCount > 0 && mom > 0) {
            gradients *= (1d - mom)
            STen.addOut(gradients, gradients, lastGradient, mom)
          }
          if (mom > 0) {
            lastGradient.copyFrom(gradients)
          }

          Scope.root { implicit scope =>
            val gradientsAsMatrix = gradients.view(gradients.shape(0), -1L)

            if (lt.shape.size == 2) {
              lt += gradientsAsMatrix.mm(gradientsAsMatrix.t)
            } else {
              lt += (gradientsAsMatrix * gradientsAsMatrix).sum(
                dim = 1,
                keepDim = false
              )
            }

            if (rt.shape.size == 2) {
              rt += (gradientsAsMatrix.t.mm(gradientsAsMatrix))
            } else {
              rt += (gradientsAsMatrix * gradientsAsMatrix).sum(
                dim = 0,
                keepDim = false
              )
            }

            val ltinv14 =
              if (
                stepCount > 100 && stepCount % updatePreconditionerEveryNIterations != 0
              )
                ltinv
              else {
                val updated = if (lt.shape.size == 2) {
                  val (u, s, vt) = lt.castToDouble.svd()
                  val s2 = (s).pow(-0.25)
                  (u * s2
                    .view(-1, 1)).mm(vt).castToType(gradients.scalarTypeByte)
                } else {
                  lt.pow(-0.25)
                }
                ltinv.copyFrom(updated)

                updated
              }

            val rtinv14 =
              if (
                stepCount > 100 && stepCount % updatePreconditionerEveryNIterations != 0
              ) rtinv
              else {
                val updated =
                  if (rt.shape.size == 2) {
                    val (u, s, vt) = rt.castToDouble.svd()
                    val s2 = (s).pow(-0.25)
                    (u * s2
                      .view(-1, 1)).mm(vt).castToType(gradients.scalarTypeByte)
                  } else {
                    rt.pow(-0.25)
                  }
                rtinv.copyFrom(updated)

                updated
              }

            val m = {
              val t1 =
                if (lt.shape.size == 2)
                  ltinv14
                    .mm(gradientsAsMatrix)
                else ltinv14.unsqueeze(1) * gradientsAsMatrix

              val t2 =
                if (rt.shape.size == 2)
                  t1.mm(rtinv14)
                else t1 * rtinv14.unsqueeze(0)

              t2.view(gradients.shape: _*)
            }

            ATen.add_out(
              param.value,
              param.value,
              m.value,
              -1 * lr
            )
          }

      }
  }
}
