package lamp.data
import aten.Tensor
import lamp.STen

sealed trait State

case class SimpleLoopState(
    model: Seq[STen],
    optimizer: Seq[STen],
    epoch: Int,
    lastValidationLoss: Option[Double],
    minValidationLoss: Option[Double],
    minValidationLossModel: Option[(Int, Seq[Tensor])],
    learningCurve: List[(Int, Double, Option[Double])]
) extends State
case class SWALoopState(
    model: Seq[STen],
    optimizer: Seq[STen],
    epoch: Int,
    lastValidationLoss: Option[Double],
    minValidationLoss: Option[Double],
    numberOfAveragedModels: Int,
    averagedModels: Option[Seq[Tensor]],
    learningCurve: List[(Int, Double, Option[Double])]
) extends State
case class SimpleThenSWALoopState(
    simple: Option[SimpleLoopState],
    swa: Option[SWALoopState]
) extends State
