package lamp.data
import aten.Tensor
import lamp.STen

sealed trait LoopState

case class SimpleLoopState(
    model: Seq[STen],
    optimizer: Seq[STen],
    epoch: Int,
    lastValidationLoss: Option[Double],
    minValidationLoss: Option[Double],
    minValidationLossModel: Option[(Int, Seq[Tensor])],
    learningCurve: List[(Int, Double, Option[(Double,Double)])]
) extends LoopState

