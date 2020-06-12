package lamp.data

import aten.Tensor
import lamp.autograd._
import aten.ATen
import aten.TensorOptions
import lamp.nn._

trait TrainingCallback {
  def apply(
      trainingLoss: Double,
      batchCount: Int,
      trainingOutput: Tensor,
      trainingTarget: Tensor
  ): Unit
}
object TrainingCallback {
  val noop = new TrainingCallback {
    def apply(
        trainingLoss: Double,
        batchCount: Int,
        trainingOutput: Tensor,
        trainingTarget: Tensor
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

sealed trait Device {
  def to(t: Tensor): Tensor
  def options: TensorOptions
}
case object CPU extends Device {
  def to(t: Tensor) = {
    t.cpu
  }
  def options: TensorOptions = TensorOptions.d.cpu
}
case class CudaDevice(i: Int) extends Device {
  assert(
    i == 0,
    "Multi gpu not implemented. Implement Tensor.to(TensorOptions)."
  )
  def to(t: Tensor): Tensor = t.cuda
  def options: TensorOptions = TensorOptions.d.cuda_index(i)
}
