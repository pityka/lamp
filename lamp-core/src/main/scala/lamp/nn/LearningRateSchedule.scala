package lamp.nn

trait LearningRateSchedule {
  def learningRateFactor(
      epoch: Long,
      lastValidationLoss: Option[Double]
  ): Double
}

object LearningRateSchedule {
  def reduceLROnPlateau(
      startFactor: Double = 1d,
      reduceFactor: Double = 0.5,
      patience: Int = 10,
      threshold: Double = 1e-4,
      relativeThresholdMode: Boolean = true,
      stopFactor: Double = 1e-4
  ) = new LearningRateSchedule {
    var min = Double.MaxValue
    var minLoc = -1L
    var activeFactor = startFactor
    def learningRateFactor(epoch: Long, lastValidationLoss: Option[Double]) = {
      require(
        lastValidationLoss.isDefined,
        "reduce lr on plateau needs validatoin loss"
      )
      val decrease =
        if (min == Double.MaxValue) lastValidationLoss.get < min
        else if (relativeThresholdMode)
          lastValidationLoss.get < min * (1d - threshold)
        else lastValidationLoss.get < (min - threshold)

      if (decrease) {
        minLoc = epoch
        min = lastValidationLoss.get
        activeFactor
      } else {
        if (epoch - minLoc >= patience) {
          activeFactor *= reduceFactor
          if (activeFactor <= stopFactor) { activeFactor = 0d }
          minLoc = epoch
          activeFactor
        } else {
          activeFactor
        }
      }
    }
  }
  def fromEpochCount(f: Long => Double) = new LearningRateSchedule {
    def learningRateFactor(
        epoch: Long,
        lastValidationLoss: Option[Double]
    ): Double =
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
      math.pow(decrementFraction, (stepCount / every).toDouble)
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
