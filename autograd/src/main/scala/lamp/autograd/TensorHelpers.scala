package lamp.autograd

import aten.Tensor
import org.saddle._
import aten.ATen
import aten.TensorOptions

object TensorHelpers {
  def unbroadcast(p: Tensor, desiredShape: List[Long]) = {
    def zip(a: List[Long], b: List[Long]) = {
      val l = b.length
      val padded = a.reverse.padTo(l, 1L).reverse
      padded.zip(b)
    }

    val sP = p.sizes.toList
    val zipped = zip(desiredShape, sP)
    val compatible = zipped.forall(x => x._1 >= x._2)

    if (compatible) p
    else {
      val dims = zipped.zipWithIndex
        .filter { case ((a, b), _) => a == 1 && b > 1 }
      val narrowed = dims.map(_._2).foldLeft(p) { (t, dim) =>
        val r = ATen.sum_1(t, Array(dim), true)
        if (t != p) {
          t.release
        }
        r
      }
      val viewed = ATen._unsafe_view(narrowed, desiredShape.toArray)
      if (narrowed != p) {
        narrowed.release
      }
      viewed
    }
  }

  def toMat(t: Tensor) = {
    val shape = t.sizes()
    if (shape.size == 2) {
      val arr = Array.ofDim[Double]((shape(0) * shape(1)).toInt)
      val data = t.copyToDoubleArray(arr)
      Mat.apply(shape(0).toInt, shape(1).toInt, arr)
    } else if (shape.size == 0) {
      val arr = Array.ofDim[Double](1)
      val data = t.copyToDoubleArray(arr)
      Mat.apply(1, 1, arr)
    } else if (shape.size == 1) {
      val arr = Array.ofDim[Double](shape(0).toInt)
      val data = t.copyToDoubleArray(arr)
      Mat.apply(1, shape(0).toInt, arr)
    } else ???
  }
  def fromMat(m: Mat[Double]) = {
    val arr = m.toArray
    val t = ATen.zeros(
      Array(m.numRows.toLong, m.numCols.toLong),
      TensorOptions.dtypeDouble
    )
    t.copyFromDoubleArray(arr)
    t
  }
}
