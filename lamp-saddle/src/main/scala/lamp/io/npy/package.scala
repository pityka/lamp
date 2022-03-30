package lamp.io

import java.nio.channels.ReadableByteChannel
import lamp.Device
import lamp.Scope
import aten.ATen
import lamp.STenOptions
import lamp.CPU
import lamp.STen
import aten.Tensor
import java.io.File

/** This package provides methods to read NPY formatted data into STen tensors
  *
  * The data is first read into to a regular JVM array, then transferred to
  * off-heap memory. The total tensor size may be larger than what a single JVM
  * array can hold.
  */
package object npy {

  def readDoubleFromChannel(
      channel: ReadableByteChannel,
      device: Device
  )(implicit
      scope: Scope
  ) = readFromChannel(7, channel, device)
  def readFloatFromChannel(
      channel: ReadableByteChannel,
      device: Device
  )(implicit
      scope: Scope
  ) = readFromChannel(6, channel, device)
  def readLongFromChannel(
      channel: ReadableByteChannel,
      device: Device
  )(implicit
      scope: Scope
  ) = readFromChannel(4, channel, device)

  def readDoubleFromFile(
      file: File,
      device: Device
  )(implicit
      scope: Scope
  ) = readFromFile(7, file, device)
  def readFloatFromFile(
      file: File,
      device: Device
  )(implicit
      scope: Scope
  ) = readFromFile(6, file, device)
  def readLongFromFile(
      file: File,
      device: Device
  )(implicit
      scope: Scope
  ) = readFromFile(4, file, device)

  def readFromFile(
      scalarType: Byte,
      file: File,
      device: Device
  )(implicit
      scope: Scope
  ) = {
    val fis = new java.io.FileInputStream(file)
    val channel = fis.getChannel
    try {
      readFromChannel(scalarType, channel, device)
    } finally {
      fis.close
    }
  }

  def readFromChannel(
      scalarType: Byte,
      channel: ReadableByteChannel,
      device: Device
  )(implicit
      scope: Scope
  ) = {
    val (dtype, topt, copy) = scalarType match {
      case 4 =>
        val dtype = org.saddle.io.npy.LongType
        val topt = STenOptions.l.value
        val copy = (ar: Array[_], offset: Long, t: Tensor) =>
          assert(
            t.copyFromLongArrayAtOffset(ar.asInstanceOf[Array[Long]], offset)
          )
        (dtype, topt, copy)
      case 6 =>
        val dtype = org.saddle.io.npy.FloatType
        val topt = STenOptions.f.value
        val copy = (ar: Array[_], offset: Long, t: Tensor) =>
          assert(
            t.copyFromFloatArrayAtOffset(ar.asInstanceOf[Array[Float]], offset)
          )
        (dtype, topt, copy)
      case 7 =>
        val dtype = org.saddle.io.npy.DoubleType
        val topt = STenOptions.d.value
        val copy = (ar: Array[_], offset: Long, t: Tensor) =>
          assert(
            t.copyFromDoubleArrayAtOffset(
              ar.asInstanceOf[Array[Double]],
              offset
            )
          )
        (dtype, topt, copy)
    }
    val (descriptor, iterator) =
      org.saddle.io.npy
        .readFromChannel(dtype, channel)
        .toOption
        .get
    assert(!descriptor.fortran, "Fortran (column-wise) layout not supported")
    val dim = descriptor.shape
    val t = ATen.zeros(
      dim.toArray,
      topt
    )
    var offset = 0L
    iterator.foreach { arr =>
      val ar = arr.toOption.get.asInstanceOf[Array[Double]]
      copy(ar, offset, t)
      offset += ar.length
    }
    assert(offset == dim.foldLeft(1L)(_ * _), "Premature end")
    if (device != CPU) {
      val t2 = device.to(t)
      t.release
      STen.owned(t2)
    } else STen.owned(t)

  }

}
