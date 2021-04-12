package lamp

import aten.Tensor

sealed trait FloatingPointPrecision {
  def convertTensor(t: Tensor): Tensor
  def convertOption[S: Sc](t: STenOptions): STenOptions
}

case object DoublePrecision extends FloatingPointPrecision {
  def convertTensor(t: Tensor): Tensor = {
    val opt = t.options().toDouble()
    val r = t.to(opt, true, true)
    opt.release
    r
  }
  def convertOption[S: Sc](t: STenOptions): STenOptions = {
    t.toDouble
  }
}
case object SinglePrecision extends FloatingPointPrecision {
  def convertTensor(t: Tensor): Tensor = {
    val opt = t.options().toFloat()
    val r = t.to(opt, true, true)
    opt.release
    r
  }
  def convertOption[S: Sc](t: STenOptions): STenOptions = {
    t.toFloat
  }
}

/** Represents a device where tensors are stored and tensor operations are executed */
sealed trait Device { self =>
  def to(t: Tensor): Tensor
  def to[S: Sc](t: STen): STen = STen.owned(self.to(t.value))
  def options[S: Sc](precision: FloatingPointPrecision): STenOptions
  def setSeed(seed: Long): Unit
}
object Device {
  def fromOptions(st: STenOptions) =
    if (st.isCPU) CPU else CudaDevice(st.deviceIndex)
}
case object CPU extends Device {
  def to(t: Tensor) = {
    t.cpu
  }
  def setSeed(seed: Long) = Tensor.manual_seed_cpu(seed)
  def options[S: Sc](precision: FloatingPointPrecision): STenOptions =
    precision.convertOption(STenOptions.d)
}
case class CudaDevice(i: Int) extends Device {
  def setSeed(seed: Long) = Tensor.manual_seed_cuda(seed, i)
  assert(
    i >= 0 && i < Tensor.getNumGPUs,
    s"Device number is wrong. Got $i. Available gpus: ${Tensor.getNumGPUs}."
  )
  def to(t: Tensor): Tensor = {
    val topt1 = t.options()
    val topt = topt1.cuda_index(i.toShort)
    val r = t.to(topt, true, true)
    topt.release
    topt1.release
    r
  }
  def options[S: Sc](precision: FloatingPointPrecision): STenOptions =
    precision.convertOption(STenOptions.d.cudaIndex(i.toShort))
}
