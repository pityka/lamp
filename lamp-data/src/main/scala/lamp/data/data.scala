package lamp.data

import scribe.Logger
import lamp.STen
import lamp.Scope

trait TrainingCallback {
  def apply(
      trainingLoss: Double,
      batchCount: Int
  ): Unit
}
object TrainingCallback {
  val noop = new TrainingCallback {
    def apply(
        trainingLoss: Double,
        batchCount: Int
    ) = ()
  }
}

trait ValidationCallback {
  def apply(
      validationOutput: STen,
      validationTarget: STen,
      validationLoss: Double,
      epochCount: Long
  ): Unit
}
object ValidationCallback {
  val noop = new ValidationCallback {
    def apply(
        validationOutput: STen,
        validationTarget: STen,
        validationLoss: Double,
        epochCount: Long
    ) = ()
  }
  def logAccuracy(logger: Logger) = new ValidationCallback {
    def apply(
        validationOutput: STen,
        validationTarget: STen,
        validationLoss: Double,
        epochCount: Long
    ): Unit = {
      val prediction = {
        Scope.leak { implicit scope =>
          validationOutput.argmax(1, false).toLongVec
        }
      }
      val corrects = prediction.zipMap(
        validationTarget.toVec
      )((a, b) => if (a == b) 1d else 0d)
      logger.info(
        s"epoch: $epochCount, validation loss: $validationLoss, corrects: ${corrects.mean}"
      )

    }
  }
}
