package lamp.util

import org.saddle._
import scala.reflect.ClassTag
import org.saddle._
import org.saddle.order._
import org.saddle.macros.BinOps._
import aten.ATen
import aten.TensorOptions
import aten.Tensor

/** INTERNAL API
  *
  * used in tests and for debugging
  */
case class NDArray[@specialized(Long, Double, Float) T](
    data: Array[T],
    shape: List[Int]
) {

  assert(data.length == shape.reduce(_ * _))
  def reshape(ns: List[Int]) = {
    assert(ns.reduce(_ * _) == shape.reduce(_ * _))
    copy(shape = ns)
  }
  def shapeOffsets = shape.drop(1).reverse.scanLeft(1)(_ * _).reverse
  def toArray = data
  def toVec(implicit st: ST[T]) = Vec(data)
  override def toString = s"NDArray(${data.deep},$shape)"
  def mapWithIndex[@specialized(Long, Double, Float) B: ClassTag](
      f: (T, List[Int]) => B
  ): NDArray[B] = {
    val k = shapeOffsets
    val arr = data.zipWithIndex.map {
      case (d, i) =>
        val idx =
          k.scanLeft((i, 0)) {
              case ((a, _), offset) =>
                val x = a / offset
                val y = a - x * offset
                (y, x)
            }
            .drop(1)
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
  )(implicit num: NUM[T], ct: ClassTag[T]): NDArray[T] = {
    assert(other.shape == this.shape) // no broadcasting
    NDArray(
      data.zip(other.data).map { case (a, b) => num.plus(a, b) } toArray,
      this.shape
    )
  }
  def -(
      other: NDArray[T]
  )(implicit num: NUM[T], ct: ClassTag[T]): NDArray[T] = {
    assert(other.shape == this.shape) // no broadcasting
    NDArray(
      data.zip(other.data).map { case (a, b) => num.minus(a, b) } toArray,
      this.shape
    )
  }
}

object NDArray {
  def zeros(shape: List[Int]) =
    NDArray(Array.ofDim[Double](shape.reduce(_ * _)), shape)
  def tensorFromNDArray(m: NDArray[Double], cuda: Boolean = false) = {
    val arr = m.toArray
    val t = ATen.zeros(
      m.shape.toArray.map(_.toLong),
      TensorOptions.dtypeDouble
    )
    t.copyFromDoubleArray(arr)
    if (cuda) {
      val t2 = t.cuda
      t.release
      t2
    } else t
  }
  def tensorFromLongNDArray(m: NDArray[Long], cuda: Boolean = false) = {
    val arr = m.toArray
    val t = ATen.zeros(
      m.shape.toArray.map(_.toLong),
      TensorOptions.dtypeLong
    )
    t.copyFromLongArray(arr)
    if (cuda) {
      val t2 = t.cuda
      t.release
      t2
    } else t
  }
  def tensorToNDArray(t: Tensor) = {
    val shape = t.sizes().toList
    val s = if (shape.size > 0) shape.reduce(_ * _).toInt else 1
    val arr = Array.ofDim[Double](s)
    val data = t.copyToDoubleArray(arr)
    NDArray(arr, if (shape.size > 0) shape.map(_.toInt) else List(1))

  }
  def tensorToLongNDArray(t: Tensor) = {
    val shape = t.sizes().toList
    val s = if (shape.size > 0) shape.reduce(_ * _).toInt else 1
    val arr = Array.ofDim[Long](s)
    val data = t.copyToLongArray(arr)
    NDArray(arr, if (shape.size > 0) shape.map(_.toInt) else List(1))

  }
}
