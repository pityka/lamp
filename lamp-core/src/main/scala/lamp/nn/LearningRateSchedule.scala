package lamp.nn

object LearningRateSchedule {
  def interpolate(
      startY: Double,
      endY: Double,
      endX: Double,
      x: Double
  ): Double = {
    val f = x / endX
    startY + (endY - startY) * f
  }
  def noop = (_: Long) => 1d
  def linear(start: Double, end: Double, maxSteps: Long) =
    (stepCount: Long) => {
      math.max(end, interpolate(start, end, maxSteps, stepCount))
    }
  def stepAfter(steps: Long, factor: Double) =
    (stepCount: Long) => {
      if (stepCount > steps) factor else 1d
    }
  def cyclicSchedule(
      maxFactor: Double,
      periodLength: Long
  ) = (stepCount: Long) => {
    val l = stepCount % periodLength
    if (l > periodLength / 2)
      maxFactor + interpolate(maxFactor, 1d, periodLength / 2, l)
    else interpolate(1d, maxFactor, periodLength / 2d, l.toDouble)
  }
}
