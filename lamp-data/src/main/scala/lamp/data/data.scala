package lamp.data

trait ValidationCallback {
  def apply(epochCount: Long, validationLoss: Double): Unit
}
object ValidationCallback {
  val noop = new ValidationCallback {
    def apply(epochCount: Long, validationLoss: Double): Unit = ()
  }
}
trait TrainingCallback {
  def apply(epochCount: Long, trainingLoss: Double): Unit
}
object TrainingCallback {
  val noop = new TrainingCallback {
    def apply(epochCount: Long, trainingLoss: Double): Unit = ()
  }
}
