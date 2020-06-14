package lamp

import aten.Tensor
import aten.TensorOptions

sealed trait FloatingPointPrecision {
  def convertTensor(t: Tensor): Tensor
  def convertOption(t: TensorOptions): TensorOptions
}

case object DoublePrecision extends FloatingPointPrecision {
  def convertTensor(t: Tensor): Tensor = {
    val opt = t.options().toDouble()
    t.to(opt, true)
  }
  def convertOption(t: TensorOptions): TensorOptions = {
    t.toDouble()
  }
}
case object SinglePrecision extends FloatingPointPrecision {
  def convertTensor(t: Tensor): Tensor = {
    val opt = t.options().toFloat()
    t.to(opt, true)
  }
  def convertOption(t: TensorOptions): TensorOptions = {
    t.toFloat()
  }
}

sealed trait Device {
  def to(t: Tensor): Tensor
  def options(precision: FloatingPointPrecision): TensorOptions
}
case object CPU extends Device {
  def to(t: Tensor) = {
    t.cpu
  }
  def options(precision: FloatingPointPrecision): TensorOptions =
    precision.convertOption(TensorOptions.d.cpu)
}
case class CudaDevice(i: Int) extends Device {
  assert(
    i == 0,
    "Multi gpu not implemented. Implement Tensor.to(TensorOptions)."
  )
  def to(t: Tensor): Tensor = t.cuda
  def options(precision: FloatingPointPrecision): TensorOptions =
    precision.convertOption(TensorOptions.d.cuda_index(i))
}
