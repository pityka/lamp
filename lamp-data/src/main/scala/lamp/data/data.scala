package lamp.data

import aten.Tensor
import lamp.autograd._
import aten.ATen
import aten.TensorOptions
import lamp.nn._

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
      validationOutput: Tensor,
      validationTarget: Tensor,
      validationLoss: Double,
      epochCount: Long
  ): Unit
}
object ValidationCallback {
  val noop = new ValidationCallback {
    def apply(
        validationOutput: Tensor,
        validationTarget: Tensor,
        validationLoss: Double,
        epochCount: Long
    ) = ()
  }
  val logAccuracy = new ValidationCallback {
    def apply(
        validationOutput: Tensor,
        validationTarget: Tensor,
        validationLoss: Double,
        epochCount: Long
    ): Unit = {
      val prediction = {
        val t = ATen.argmax(validationOutput, 1, false)
        val r = TensorHelpers
          .toMatLong(t)
          .toVec
        t.release
        r
      }
      val corrects = prediction.zipMap(
        TensorHelpers.toMatLong(validationTarget).toVec
      )((a, b) => if (a == b) 1d else 0d)
      scribe.info(
        s"epoch: $epochCount, validation loss: $validationLoss, corrects: ${corrects.mean}"
      )

    }
  }
}
