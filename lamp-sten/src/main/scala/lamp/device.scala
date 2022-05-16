package lamp

import aten.Tensor

sealed trait FloatingPointPrecision {
  def convertTensor(t: Tensor): Tensor
  def convertOption[S: Sc](t: STenOptions): STenOptions
  def scalarTypeByte: Byte
}

case object DoublePrecision extends FloatingPointPrecision {
  val scalarTypeByte = 7
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
  val scalarTypeByte = 6
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
case object HalfPrecision extends FloatingPointPrecision {
  val scalarTypeByte = 5
  def convertTensor(t: Tensor): Tensor = {
    aten.ATen._cast_Half(t, true)
  }
  def convertOption[S: Sc](t: STenOptions): STenOptions = {
    // t.toHalf
    ???
  }
}

/** Represents a device where tensors are stored and tensor operations are
  * executed
  */
sealed trait Device { self =>
  def to(t: Tensor): Tensor
  def to[S: Sc](t: STen): STen = STen.owned(self.to(t.value))

  /** Copies tensors to this device in a single cross device copy. srcBuffer is
    * supposed to be on the source device. dstBuffer has to be on `this` device.
    * Tensors are first copied to the srcBuffer, then the srcBuffer is copied to
    * dstBuffer, then the dstBuffer is split into views.
    *
    * All tensors must have the same data type.
    *
    * Might make sense to pin the srcBuffer.
    *
    * @param tensors
    * @param srcBuffer
    * @param dstBuffer
    * @return
    */
  def toBatched[S: Sc](
      tensors: Seq[STen],
      buffers: BufferPair
  ): Seq[STen] = self
    .toBatchedImpl(
      tensors.map(_.value),
      buffers.source.value,
      buffers.destination.value
    )
    .map(t => STen.owned(t))

  def allocateBuffers[S: Sc](size: Long, options: STenOptions) =
    BufferPair.allocate(size, this, options)

  private def toBatchedImpl(
      tensors: Seq[Tensor],
      hostBuffer: Tensor,
      deviceBuffer: Tensor
  ): Seq[Tensor] = {
    val views = tensors.map(t => aten.ATen._unsafe_view(t, Array(-1L)))
    val viewSizes = views.map(_.numel())
    val shapes = tensors.map(_.sizes)

    aten.ATen.cat_out(hostBuffer, views.toArray, 0)
    views.foreach(_.release)

    deviceBuffer.copyFrom(hostBuffer, true)

    val viewedBuffer =
      aten.ATen._unsafe_view(deviceBuffer, Array(viewSizes.sum))

    val offsets = viewSizes.scanLeft(0L)(_ + _).dropRight(1)

    val slices = offsets.zip(viewSizes).map { case (offset, size) =>
      aten.ATen.narrow_0(viewedBuffer, 0, offset, size)
    }
    viewedBuffer.release

    val r = slices.zip(shapes).toSeq.map { case (slice, shape) =>
      aten.ATen._unsafe_view(slice, shape)
    }
    val clones = r.map { t =>
      aten.ATen.clone(t)
    }

    slices.foreach(_.release())
    clones
  }
  def to[S: Sc](t: STenOptions): STenOptions
  def options[S: Sc](precision: FloatingPointPrecision): STenOptions
  def setSeed(seed: Long): Unit

  /** Executes f on a new stream f must not switch to other threads Restores the
    * stream to the default stream
    */
  def withOtherStreamThenSync[A](synchronizeBefore: Boolean)(f: => A): A

  def measureTime[A](f: => A): (A, Long)
}
object Device {
  def fromOptions(st: STenOptions) =
    if (st.isCPU) CPU else CudaDevice(st.deviceIndex)
}
case object CPU extends Device {
  def measureTime[A](f: => A): (A, Long) = {
    val t1 = System.nanoTime
    val r = f
    val t2 = System.nanoTime
    (r, t2 - t1)
  }
  def withOtherStreamThenSync[A](synchronizeBefore: Boolean)(f: => A): A = f
  def to[S: Sc](t: STenOptions): STenOptions = t.cpu
  def to(t: Tensor) = {
    t.cpu
  }
  def setSeed(seed: Long) = Tensor.manual_seed_cpu(seed)
  def options[S: Sc](precision: FloatingPointPrecision): STenOptions =
    precision.convertOption(STenOptions.d)

}
case class CudaDevice(i: Int) extends Device {

  def measureTime[A](f: => A): (A, Long) = {
    val default = aten.CudaStream.getDefaultCUDAStream(i.toByte)
    val t1 = System.nanoTime()
    val r = f
    default.synchronize()
    val t2 = System.nanoTime
    (r, t2 - t1)
  }

  def withOtherStreamThenSync[A](synchronizeBefore: Boolean)(f: => A): A = {
    val default = aten.CudaStream.getDefaultCUDAStream(i.toByte)
    if (synchronizeBefore) { default.synchronize() }
    val other = aten.CudaStream.getStreamFromPool(false, i.toByte)
    aten.CudaStream.setCurrentCUDAStream(other)
    val a = f
    other.synchronize()
    aten.CudaStream.setCurrentCUDAStream(default)
    a
  }
  def to[S: Sc](t: STenOptions): STenOptions = t.cudaIndex(i.toShort)
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

case class BufferPair(
    source: STen,
    destination: STen
)

object BufferPair {
  implicit val movable: Movable[BufferPair] =
    Movable.by(b => (b.source, b.destination))
  def allocate(size: Long, device: Device, options: STenOptions)(implicit
      scope: Scope
  ) = Scope { implicit scope =>
    val host = STen.zeros(List(size), CPU.to(options)).pin
    val d = device.to(host)
    BufferPair(host, d)

  }
}
