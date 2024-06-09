package lamp.data
import cats.effect.IO
trait ValidationCallback[M] {
  def apply(epochCount: Long, validationLoss: Double, model: M): IO[Unit]
}

trait TrainingCallback[M] {
  def apply(epochCount: Long, trainingLoss: Double, model: M): IO[Unit]
}
