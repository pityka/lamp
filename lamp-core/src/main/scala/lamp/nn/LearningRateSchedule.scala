package lamp.nn

trait LearningRateSchedule {
  def factor(epoch: Long, lastValidationLoss: Option[Double]): Double
}

object LearningRateSchedule {
  def fromEpochCount(f: Long => Double) = new LearningRateSchedule {
    def factor(epoch: Long, lastValidationLoss: Option[Double]): Double =
      f(epoch)
  }
  def interpolate(
      startY: Double,
      endY: Double,
      endX: Double,
      x: Double
  ): Double = {
    val f = x / endX
    startY + (endY - startY) * f
  }
  def noop = fromEpochCount((_: Long) => 1d)
  def decrement(every: Int, decrementFraction: Double) =
    fromEpochCount((stepCount: Long) =>
      math.pow(1d - decrementFraction, (stepCount / every).toDouble)
    )
  def linear(start: Double, end: Double, maxSteps: Long) =
    fromEpochCount((stepCount: Long) => {
      math.max(end, interpolate(start, end, maxSteps, stepCount))
    })
  def stepAfter(steps: Long, factor: Double) =
    fromEpochCount((stepCount: Long) => {
      if (stepCount > steps) factor else 1d
    })
  def cyclicSchedule(
      maxFactor: Double,
      periodLength: Long
  ) =
    fromEpochCount((stepCount: Long) => {
      val l = stepCount % periodLength
      if (l > periodLength / 2)
        maxFactor + interpolate(maxFactor, 1d, periodLength / 2, l)
      else interpolate(1d, maxFactor, periodLength / 2d, l.toDouble)
    })
}
