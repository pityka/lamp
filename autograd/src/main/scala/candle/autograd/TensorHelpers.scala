package candle.autograd

import aten.Tensor
import org.saddle._
import aten.ATen
import aten.TensorOptions

object TensorHelpers {
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
