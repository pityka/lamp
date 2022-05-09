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
case class SWALoopState(
    model: Seq[STen],
    optimizer: Seq[STen],
    epoch: Int,
    lastValidationLoss: Option[Double],
    minValidationLoss: Option[Double],
    numberOfAveragedModels: Int,
    averagedModels: Option[Seq[Tensor]],
    learningCurve: List[(Int, Double, Option[Double])]
) extends LoopState
case class SimpleThenSWALoopState(
    simple: SimpleLoopState,
    swa: Option[SWALoopState]
) extends LoopState
