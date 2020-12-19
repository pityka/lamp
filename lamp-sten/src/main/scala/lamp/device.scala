package lamp

import aten.Tensor

sealed trait FloatingPointPrecision {
  def convertTensor(t: Tensor): Tensor
  def convertOption[S: Sc](t: STenOptions): STenOptions
}

case object DoublePrecision extends FloatingPointPrecision {
  def convertTensor(t: Tensor): Tensor = {
    val opt = t.options().toDouble()
    t.to(opt, true, true)
  }
  def convertOption[S: Sc](t: STenOptions): STenOptions = {
    t.toDouble
  }
}
case object SinglePrecision extends FloatingPointPrecision {
  def convertTensor(t: Tensor): Tensor = {
    val opt = t.options().toFloat()
    t.to(opt, true, true)
  }
  def convertOption[S: Sc](t: STenOptions): STenOptions = {
    t.toFloat
  }
}

sealed trait Device { self =>
  def to(t: Tensor): Tensor
  def to[S: Sc](t: STen): STen = STen.owned(self.to(t.value))
  def options[S: Sc](precision: FloatingPointPrecision): STenOptions
}
case object CPU extends Device {
  def to(t: Tensor) = {
    t.cpu
  }
  def options[S: Sc](precision: FloatingPointPrecision): STenOptions =
    precision.convertOption(STenOptions.d)
}
case class CudaDevice(i: Int) extends Device {
  assert(
    i >= 0 && i < Tensor.getNumGPUs,
    s"Device number is wrong. Got $i. Available gpus: ${Tensor.getNumGPUs}."
  )
  def to(t: Tensor): Tensor = {
    val topt = t.options().cuda_index(i.toShort)
    t.to(topt, true, true)
  }
  def options[S: Sc](precision: FloatingPointPrecision): STenOptions =
    precision.convertOption(STenOptions.d.cudaIndex(i.toShort))
}
