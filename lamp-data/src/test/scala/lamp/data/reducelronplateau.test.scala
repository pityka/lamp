package lamp.data

import org.scalatest.funsuite.AnyFunSuite
import lamp.nn.LearningRateSchedule

class ReduceLROnPlateauSuite extends AnyFunSuite {
  test("reduce lr test") {
    val lrs = LearningRateSchedule.reduceLROnPlateau(stopFactor = 0.1)
    assert(lrs.learningRateFactor(0L, Some(10d)) == 1d)
    assert(lrs.learningRateFactor(1L, Some(10d)) == 1d)
    assert(lrs.learningRateFactor(9L, Some(10d)) == 1d)
    assert(lrs.learningRateFactor(10L, Some(10d)) == 0.5)
    assert(lrs.learningRateFactor(18L, Some(11d)) == 0.5)
    assert(lrs.learningRateFactor(19L, Some(9d)) == 0.5)
    assert(lrs.learningRateFactor(28L, Some(9d - 1e-6)) == 0.5)
    assert(lrs.learningRateFactor(29L, Some(9d - 1e-6)) == 0.25)
    assert(lrs.learningRateFactor(39L, Some(9d - 1e-6)) == 0.125)
    assert(lrs.learningRateFactor(50L, Some(9d - 1e-6)) == 0.0)
  }
}
