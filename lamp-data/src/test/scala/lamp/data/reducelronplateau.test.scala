package lamp.data

import org.scalatest.funsuite.AnyFunSuite
import lamp.nn.LearningRateSchedule

class ReduceLROnPlateauSuite extends AnyFunSuite {
  test("reduce lr test") {
    val lrs = LearningRateSchedule.reduceLROnPlateau(stopFactor = 0.1)
    val st = lrs.init

    import cats.data.State
    def next(e: Long, d: Double, exp: Double) =
      State(st => lrs.learningRateFactor(st, e, Some(d))).map { b =>
        assert(b == exp)
        b
      }

    (for {
      _ <- next(0L, 10d, 1d)
      _ <- next(1L, 10d, 1d)
      _ <- next(9L, 10d, 1d)
      _ <- next(10L, 10d, 0.5)
      _ <- next(18L, 11d, 0.5)
      _ <- next(19L, 9d, 0.5)
      _ <- next(28L, 9d - 1e-6, 0.5)
      _ <- next(29L, 9d - 1e-6, 0.25)
      _ <- next(39L, 9d - 1e-6, 0.125)
      _ <- next(50, 9d - 1e-6, 0.0)
    } yield ()).run(st).value

  }
}
