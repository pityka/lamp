package lamp.util

import scala.reflect.ClassTag
// import org.saddle._
import aten.ATen
import aten.Tensor
import lamp.STenOptions
import lamp.EmptyMovable
import lamp.Movable

/** INTERNAL API
  *
  * used in tests and for debugging
  */
private[lamp] case class NDArray[@specialized(Long, Double, Float) T](
    data: Array[T],
    shape: List[Int]
) {

  assert(
    data.length == shape.foldLeft(1)(_ * _),
    "Dimension does not match up with elements."
  )
  def reshape(ns: List[Int]) = {
    assert(ns.reduce(_ * _) == shape.reduce(_ * _))
    copy(shape = ns)
  }
  def shapeOffsets = shape.drop(1).reverse.scanLeft(1)(_ * _).reverse
  def toArray = data

  override def toString = s"NDArray(${data.toVector},$shape)"
  def mapWithIndex[@specialized(Long, Double, Float) B: ClassTag](
      f: (T, List[Int]) => B
  ): NDArray[B] = {
    val k = shapeOffsets
    val arr = data.zipWithIndex.map { case (d, i) =>
      val idx =
        k.scanLeft((i, 0)) { case ((a, _), offset) =>
          val x = a / offset
          val y = a - x * offset
          (y, x)
        }.drop(1)
          .map(_._2)
      f(d, idx)
    }
    NDArray(arr.toArray, shape)
  }
  def set(idx: List[Int], v: T) = {
    val off: Int = idx.zip(shapeOffsets).map(a => a._1 * a._2).sum
    data(off) = v
  }
  def +(
      other: NDArray[T]
  )(implicit num: Numeric[T], ct: ClassTag[T]): NDArray[T] = {
    assert(other.shape == this.shape) // no broadcasting
    NDArray(
      data.zip(other.data).map { case (a, b) => num.plus(a, b) } toArray,
      this.shape
    )
  }
  def -(
      other: NDArray[T]
  )(implicit num: Numeric[T], ct: ClassTag[T]): NDArray[T] = {
    assert(other.shape == this.shape) // no broadcasting
    NDArray(
      data.zip(other.data).map { case (a, b) => num.minus(a, b) } toArray,
      this.shape
    )
  }
}

private[lamp] object NDArray {
  def zeros(shape: List[Int]) =
    NDArray(Array.ofDim[Double](shape.foldLeft(1)(_ * _)), shape)
  def tensorFromNDArray(m: NDArray[Double], device: lamp.Device = lamp.CPU) = {
    val arr = m.toArray
    val t = ATen.zeros(
      m.shape.toArray.map(_.toLong),
      STenOptions.d.value
    )
    val success = t.copyFromDoubleArray(arr)
    if (!success) {
      throw new RuntimeException("Failed to copy")
    }
    if (device.isInstanceOf[lamp.CudaDevice]) {
      val t2 = t.cuda
      t.release
      t2
    } else if (device == lamp.MPS) {
      val t3 = aten.ATen._cast_Float(t,false)
      val t2 = device.to(t3)
      t.release
      t3.release()
      t2
    } else t
  }
  def tensorFromLongNDArray(
      m: NDArray[Long],
      device: lamp.Device = lamp.CPU
  ) = {
    val arr = m.toArray
    val t = ATen.zeros(
      m.shape.toArray.map(_.toLong),
      STenOptions.l.value
    )
    val success = t.copyFromLongArray(arr)
    if (!success) {

      throw new RuntimeException("Failed to copy")
    }
    if (device.isInstanceOf[lamp.CudaDevice]) {
      val t2 = t.cuda
      t.release
      t2
    } else if (device == lamp.MPS) {
      val t2 = device.to(t)
      t.release
      t2
    } else t
  }
  def tensorToNDArray(t0: Tensor) = {
    val t = if (t0.isCuda || t0.isMps) t0.cpu else t0
    val op = t.options()
    try {
      val shape = t.sizes().toList
      val s = if (shape.size > 0) shape.reduce(_ * _).toInt else 1
      val arr = Array.ofDim[Double](s)
      val data =
        if (op.isDouble())
          t.copyToDoubleArray(arr)
        else {
          val t3 = aten.ATen._cast_Double(t, false)
          val b = t3.copyToDoubleArray(arr)
          t3.release
          b
        }

      if (!data) {
        throw new RuntimeException("Failed to copy")
      }
      NDArray(arr, if (shape.size > 0) shape.map(_.toInt) else List(1))
    } finally {
      op.release
      if (t != t0) { t.release }
    }
  }
  def tensorToLongNDArray(t0: Tensor) = {
    val t = if (t0.isCuda || t0.isMps) t0.cpu else t0
    try {
      val shape = t.sizes().toList
      val s = if (shape.size > 0) shape.reduce(_ * _).toInt else 1
      val arr = Array.ofDim[Long](s)
      val data = t.copyToLongArray(arr)
      if (!data) {
        throw new RuntimeException(s"Failed to copy $t")
      }
      NDArray(arr, if (shape.size > 0) shape.map(_.toInt) else List(1))
    } finally {
      if (t != t0) { t.release }
    }
  }
  def tensorToFloatNDArray(t0: Tensor) = {
    val t = if (t0.isCuda || t0.isMps) t0.cpu else t0
    try {
      val shape = t.sizes().toList
      val s = if (shape.size > 0) shape.reduce(_ * _).toInt else 1
      val arr = Array.ofDim[Float](s)
      val data = t.copyToFloatArray(arr)
      if (!data) {
        throw new RuntimeException("Failed to copy")
      }
      NDArray(arr, if (shape.size > 0) shape.map(_.toInt) else List(1))
    } finally {
      if (t != t0) { t.release }
    }
  }

  @scala.annotation.nowarn
  implicit def m[T: EmptyMovable]: EmptyMovable[NDArray[T]] = Movable.empty
}
