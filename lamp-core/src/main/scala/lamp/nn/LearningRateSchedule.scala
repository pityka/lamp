package lamp.nn

trait LearningRateSchedule[State] {
  def init: State
  def learningRateFactor(
      state: State,
      epoch: Long,
      lastValidationLoss: Option[Double]
  ): (State, Double)
}

object LearningRateSchedule {
  case class ReduceLROnPlateauState(
      min: Double = Double.MaxValue,
      minLoc: Long = -1L,
      activeFactor: Double
  )
  def reduceLROnPlateau(
      startFactor: Double = 1d,
      reduceFactor: Double = 0.5,
      patience: Int = 10,
      threshold: Double = 1e-4,
      relativeThresholdMode: Boolean = true,
      stopFactor: Double = 1e-4
  ) = new LearningRateSchedule[ReduceLROnPlateauState] {

    def init = ReduceLROnPlateauState(
      min = Double.MaxValue,
      minLoc = -1L,
      activeFactor = startFactor
    )

    def learningRateFactor(
        state: ReduceLROnPlateauState,
        epoch: Long,
        lastValidationLoss: Option[Double]
    ) = {
      require(
        lastValidationLoss.isDefined,
        "reduce lr on plateau needs validatoin loss"
      )
      val decrease =
        if (state.min == Double.MaxValue) lastValidationLoss.get < state.min
        else if (relativeThresholdMode)
          lastValidationLoss.get < state.min * (1d - threshold)
        else lastValidationLoss.get < (state.min - threshold)

      if (decrease) {

        (
          state.copy(minLoc = epoch, min = lastValidationLoss.get),
          state.activeFactor
        )
      } else {
        if (epoch - state.minLoc >= patience) {
          var x = state.activeFactor * reduceFactor
          if (x <= stopFactor) { x = 0d }
          (state.copy(activeFactor = x, minLoc = epoch), x)
        } else {
          (state, state.activeFactor)
        }
      }
    }
  }
  def fromEpochCount(f: Long => Double) = new LearningRateSchedule[Unit] {
    def init = ()
    def learningRateFactor(
        state: Unit,
        epoch: Long,
        lastValidationLoss: Option[Double]
    ): (Unit, Double) =
      ((), f(epoch))
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
      math.max(
        end,
        interpolate(start, end, maxSteps.toDouble, stepCount.toDouble)
      )
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
        maxFactor + interpolate(
          maxFactor,
          1d,
          (periodLength / 2).toDouble,
          l.toDouble
        )
      else interpolate(1d, maxFactor, periodLength / 2d, l.toDouble)
    })
}
