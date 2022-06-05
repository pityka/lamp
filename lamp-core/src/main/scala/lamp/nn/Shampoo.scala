package lamp.nn

import aten.ATen
import lamp.STen
import lamp.Scope
import lamp.STenOptions

object Shampoo {
  def factory(
      learningRate: OptimizerHyperparameter = simple(0.001),
      clip: Option[Double] = None,
      eps: Double = 1e-4
  ) =
    (parameters: Seq[(STen, PTag)]) =>
      Shampoo(
        parameters,
        learningRate,
        clip,
        eps
      )
}

/** @see
  *   https://arxiv.org/pdf/1802.09568.pdf Algorithm 1
  */
case class Shampoo(
    parameters: Seq[(STen, PTag)],
    learningRate: OptimizerHyperparameter = simple(0.001),
    clip: Option[Double] = None,
    eps: Double = 1e-4
) extends Optimizer {
  val scope = Scope.free
  val lt: List[STen] = parameters.toList.map { case (param, _) =>
    val dim1 = param.shape(0)
    val t = STen.eye(dim1.toInt, param.options(scope))(scope)
    t *= eps
    t
  }
  val rt: List[STen] = parameters.toList.map { case (param, _) =>
    val dim2 = param.shape.drop(1).foldLeft(1L)(_ * _)
    val t = STen.eye(dim2.toInt, param.options(scope))(scope)
    t *= eps
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
  def state = List(stepCountSTen) ++ lt ++ rt
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
      .filter(_._1._1._2.isDefined)
      .foreach {
        case ((((_, _), None), _), _) =>
          // won't happent, see filter above
          ???
        case ((((param, tag), Some(gradients)), lt), rt) =>
          val lr = learningRate(tag)

          Scope.root { implicit sc =>
            val gradientsAsMatrix = gradients.view(gradients.shape(0), -1L)
            lt += gradientsAsMatrix.mm(gradientsAsMatrix.t(scope))

            rt += (gradientsAsMatrix.t.mm(gradientsAsMatrix))

            val ltinv14 = {
              val (u, s, vt) = lt.castToDouble.svd()
              val s2 = (s).pow(-0.25)
              (u * s2.view(-1, 1)).mm(vt).castToType(gradients.scalarTypeByte)
            }

            val rtinv14 = {
              val (u, s, vt) = rt.castToDouble.svd(false)
              val s2 = (s).pow(-0.25)
              (u * s2.view(-1, 1)).mm(vt).castToType(gradients.scalarTypeByte)
            }

            val m = ltinv14
              .mm(gradientsAsMatrix)
              .mm(rtinv14)
              .view(gradients.shape: _*)

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
