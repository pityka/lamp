package lamp.nn

import lamp.autograd.{Variable, param, const}
import aten.ATen
import aten.TensorOptions
import aten.Tensor
import lamp.syntax

case class Meanshift(
    means: Variable,
    dim: List[Int],
    runningMeanMomentum: Double,
    training: Boolean,
    var runningMean: Option[Tensor]
) extends Module {

  override def asTraining = copy(training = true)
  override def asEval = copy(training = false)

  override def load(parameters: Seq[Tensor]) = {
    copy(means = param(parameters.head))
  }
  override def parameters: Seq[(Variable, PTag)] =
    List(means -> Meanshift.Means)
  def forward(x: Variable): Variable = {
    val mean = if (training) x.mean(dim) else const(runningMean.get)
    if (training) {
      if (runningMean.isEmpty) {
        runningMean = Some(ATen.clone(mean.value))
      } else {
        val m = runningMean.get
        m *= runningMeanMomentum
        m.addcmul(mean.value, (1d - runningMeanMomentum))
      }
    }
    (x - mean) + means
  }
}

object Meanshift {
  case object Means extends LeafTag
  def apply(
      size: List[Long],
      dim: List[Int] = List(0),
      tOpt: TensorOptions = TensorOptions.dtypeDouble,
      runningMeanMomentum: Double = 0.1,
      training: Boolean = true
  ): Meanshift =
    Meanshift(
      dim = dim,
      means = param(ATen.zeros(size.toArray, tOpt)),
      runningMeanMomentum = runningMeanMomentum,
      training = training,
      runningMean = None
    )
}
