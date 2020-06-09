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
