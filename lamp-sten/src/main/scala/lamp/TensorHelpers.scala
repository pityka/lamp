package lamp

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

    if (compatible) {
      ATen._unsafe_view(p, desiredShape.toArray)
    } else {
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
    if (t.options.scalarTypeByte() == 6) toFloatMat(t).map(_.toDouble)
    else {
      assert(
        t.options.scalarTypeByte == 7,
        s"Expected Double Tensor. Got scalartype: ${t.options.scalarTypeByte}"
      )
      val shape = t.sizes()
      if (shape.size == 2) {
        val arr = Array.ofDim[Double]((shape(0) * shape(1)).toInt)
        assert(t.copyToDoubleArray(arr), "failed to copy")
        Mat.apply(shape(0).toInt, shape(1).toInt, arr)
      } else if (shape.size == 0) {
        val arr = Array.ofDim[Double](1)
        assert(t.copyToDoubleArray(arr))
        Mat.apply(1, 1, arr)
      } else if (shape.size == 1) {
        val arr = Array.ofDim[Double](shape(0).toInt)
        assert(t.copyToDoubleArray(arr))
        Mat.apply(1, shape(0).toInt, arr)
      } else throw new RuntimeException("shape: " + shape.deep)
    }
  }
  def toVec(t: Tensor) = {
    if (t.options().scalarTypeByte() == 6) {
      assert(
        t.numel <= Int.MaxValue,
        "Tensor too long to fit into a java array"
      )
      val arr = Array.ofDim[Float](t.numel.toInt)
      assert(t.copyToFloatArray(arr))
      arr.toVec.map(_.toDouble)
    } else {
      assert(
        t.options.scalarTypeByte == 7,
        s"Expected Double Tensor. Got scalartype: ${t.options.scalarTypeByte}"
      )
      assert(
        t.numel <= Int.MaxValue,
        "Tensor too long to fit into a java array"
      )
      val arr = Array.ofDim[Double](t.numel.toInt)
      assert(t.copyToDoubleArray(arr))
      arr.toVec
    }
  }
  def toLongVec(t: Tensor) = {
    assert(
      t.options.scalarTypeByte == 4,
      s"Expected Double Tensor. Got scalartype: ${t.options.scalarTypeByte}"
    )
    assert(t.numel <= Int.MaxValue, "Tensor too long to fit into a java array")
    val arr = Array.ofDim[Long](t.numel.toInt)
    assert(t.copyToLongArray(arr))
    arr.toVec
  }
  def toFloatMat(t: Tensor) = {
    assert(
      t.options.scalarTypeByte == 6,
      s"Expected Double Tensor. Got scalartype: ${t.options.scalarTypeByte}"
    )
    val shape = t.sizes()
    if (shape.size == 2) {
      val arr = Array.ofDim[Float]((shape(0) * shape(1)).toInt)
      assert(t.copyToFloatArray(arr))
      Mat.apply(shape(0).toInt, shape(1).toInt, arr)
    } else if (shape.size == 0) {
      val arr = Array.ofDim[Float](1)
      assert(t.copyToFloatArray(arr))
      Mat.apply(1, 1, arr)
    } else if (shape.size == 1) {
      val arr = Array.ofDim[Float](shape(0).toInt)
      assert(t.copyToFloatArray(arr))
      Mat.apply(1, shape(0).toInt, arr)
    } else throw new RuntimeException("shape: " + shape.deep)
  }
  def toLongMat(t: Tensor) = {
    assert(
      t.options.scalarTypeByte == 4,
      s"Expected Long Tensor. Got scalartype: ${t.options.scalarTypeByte}"
    )
    val shape = t.sizes()
    if (shape.size == 2) {
      val arr = Array.ofDim[Long]((shape(0) * shape(1)).toInt)
      assert(t.copyToLongArray(arr))
      Mat.apply(shape(0).toInt, shape(1).toInt, arr)
    } else if (shape.size == 0) {
      val arr = Array.ofDim[Long](1)
      assert(t.copyToLongArray(arr))
      Mat.apply(1, 1, arr)
    } else if (shape.size == 1) {
      val arr = Array.ofDim[Long](shape(0).toInt)
      assert(t.copyToLongArray(arr))
      Mat.apply(1, shape(0).toInt, arr)
    } else ???
  }
  def fromMatList(
      m: Seq[Mat[Double]],
      cuda: Boolean = false
  ): Tensor =
    fromMatList(
      m,
      device = if (cuda) CudaDevice(0) else CPU,
      precision = DoublePrecision
    )
  def fromMatList(
      m: Seq[Mat[Double]],
      device: Device,
      precision: FloatingPointPrecision
  ) = {
    assert(m.map(_.numRows).distinct.size == 1, "shapes must be homogeneous")
    assert(m.map(_.numCols).distinct.size == 1, "shapes must be homogeneous")
    val arr: Array[Double] = m.toArray.flatMap(_.toArray)
    val d1 = m.headOption.map(_.numRows).getOrElse(0)
    val d2 = m.headOption.map(_.numCols).getOrElse(0)
    val t = ATen.zeros(
      Array(m.length, d1, d2),
      TensorOptions.dtypeDouble
    )
    assert(t.copyFromDoubleArray(arr))
    if (device != CPU || precision != DoublePrecision) {
      val t2 = t.to(device.options(precision), true)
      t.release
      t2
    } else t
  }
  def fromMat(
      m: Mat[Double],
      cuda: Boolean = false
  ): Tensor =
    fromMat(
      m,
      precision = DoublePrecision,
      device = if (cuda) CudaDevice(0) else CPU
    )
  def fromFloatMat(
      m: Mat[Float],
      device: Device
  ) = {
    val arr = m.toArray
    val t = ATen.zeros(
      Array(m.numRows.toLong, m.numCols.toLong),
      TensorOptions.dtypeFloat()
    )
    assert(t.copyFromFloatArray(arr))
    if (device != CPU) {
      val t2 = t.to(device.options(SinglePrecision), true)
      t.release
      t2
    } else t
  }
  def fromMat(
      m: Mat[Double],
      device: Device,
      precision: FloatingPointPrecision
  ) = {
    val arr = m.toArray
    val t = ATen.zeros(
      Array(m.numRows.toLong, m.numCols.toLong),
      TensorOptions.dtypeDouble
    )
    assert(t.copyFromDoubleArray(arr))
    if (device != CPU || precision != DoublePrecision) {
      val t2 = t.to(device.options(precision), true)
      t.release
      t2
    } else t
  }
  def fromLongMat(m: Mat[Long], cuda: Boolean = false): Tensor =
    fromLongMat(m, device = if (cuda) CudaDevice(0) else CPU)
  def fromLongMat(m: Mat[Long], device: Device) = {
    val arr = m.toArray
    val t = ATen.zeros(
      Array(m.numRows.toLong, m.numCols.toLong),
      TensorOptions.dtypeLong
    )
    assert(t.copyFromLongArray(arr))
    if (device != CPU) {
      val t2 = device.to(t)
      t.release
      t2
    } else t
  }
  def fromLongVec(m: Vec[Long], cuda: Boolean = false): Tensor =
    fromLongVec(m, device = if (cuda) CudaDevice(0) else CPU)
  def fromLongVec(m: Vec[Long], device: Device) = {
    val arr = m.toArray
    val t = ATen.zeros(
      Array(m.length.toLong),
      TensorOptions.dtypeLong
    )
    assert(t.copyFromLongArray(arr))
    if (device != CPU) {
      val t2 = device.to(t)
      t.release
      t2
    } else t
  }
  def fromVec(
      m: Vec[Double],
      cuda: Boolean = false
  ): Tensor =
    fromVec(
      m,
      precision = DoublePrecision,
      device = if (cuda) CudaDevice(0) else CPU
    )
  def fromVec(
      m: Vec[Double],
      device: Device,
      precision: FloatingPointPrecision
  ) = {
    val arr = m.toArray
    val t = ATen.zeros(
      Array(m.length.toLong),
      TensorOptions.dtypeDouble
    )
    assert(t.copyFromDoubleArray(arr))
    if (device != CPU || precision != DoublePrecision) {
      val t2 = t.to(device.options(precision), true)
      t.release
      t2
    } else t
  }

  def device(t: Tensor) = {
    val op = t.options
    if (op.isCPU) CPU
    else CudaDevice(op.deviceIndex)
  }

  def precision(t: Tensor) = {
    val op = t.options
    if (op.isFloat()) Some(SinglePrecision)
    else if (op.isDouble) Some(DoublePrecision)
    else None
  }

}
