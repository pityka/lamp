package lamp.data

import scribe.Logger
import lamp.STen
import lamp.Scope

trait TrainingBatchCallback {
  def apply(
      trainingLoss: Double,
      batchCount: Int
  ): Unit
}
object TrainingBatchCallback {
  val noop = new TrainingBatchCallback {
    def apply(
        trainingLoss: Double,
        batchCount: Int
    ) = ()
  }
}

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

trait ValidationBatchCallback {
  def apply(
      validationOutput: STen,
      validationTarget: STen,
      validationLoss: Double,
      epochCount: Long,
      batchCount: Long
  ): Unit
}
object ValidationBatchCallback {
  val noop = new ValidationBatchCallback {
    def apply(
        validationOutput: STen,
        validationTarget: STen,
        validationLoss: Double,
        epochCount: Long,
        batchCount: Long
    ) = ()
  }
  def logAccuracy(logger: Logger) = new ValidationBatchCallback {
    def apply(
        validationOutput: STen,
        validationTarget: STen,
        validationLoss: Double,
        epochCount: Long,
        batchCount: Long
    ): Unit = {
      val prediction = {
        Scope.leak { implicit scope =>
          validationOutput.argmax(1, false).toLongVec
        }
      }
      val corrects = prediction.zipMap(
        validationTarget.toLongVec
      )((a, b) => if (a == b) 1d else 0d)
      logger.info(
        s"epoch: $epochCount, validation loss: $validationLoss, corrects: ${corrects.mean}"
      )

    }
  }
}
