package lamp.data

trait ValidationCallback[M] {
  def apply(epochCount: Long, validationLoss: Double, model: M): Unit
}

trait TrainingCallback[M] {
  def apply(epochCount: Long, trainingLoss: Double, model: M): Unit
}
