package lamp.data.distributed

import lamp.Movable
import lamp.EmptyMovable

case class LoopState(
    epoch: Int,
    lastValidationLoss: Option[Double],
    minValidationLoss: Option[Double],
    minValidationEpoch: Option[Int],
    learningCurve: List[(Int, Double, Option[(Double, Double)])]
)
object LoopState {
  implicit val movable: EmptyMovable[LoopState] = Movable.empty
}
