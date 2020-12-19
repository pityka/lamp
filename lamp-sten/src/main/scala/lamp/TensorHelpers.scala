package lamp

import aten.Tensor
import org.saddle._
import aten.ATen

object TensorHelpers {
  def unbroadcast(p: Tensor, desiredShape: List[Long]): Option[Tensor] =
    if (desiredShape == p.sizes.toList) None
    else {
      def zip(a: List[Long], b: List[Long]) = {
        val l = b.length
        val padded = a.reverse.padTo(l, 1L).reverse
        padded.zip(b)
      }

      val sP = p.sizes.toList
      val zipped = zip(desiredShape, sP)
      val compatible = zipped.forall(x => x._1 >= x._2)

      if (compatible) {
        Some(ATen._unsafe_view(p, desiredShape.toArray))
      } else {
        val dims = zipped.zipWithIndex
          .filter { case ((a, b), _) => a == 1 && b > 1 }
        val narrowed =
          if (dims.isEmpty) p
          else {
            ATen.sum_1(p, dims.map(_._2.toLong).toArray, true)
          }

        val viewed =
          if (narrowed.sizes != desiredShape.toArray)
            ATen._unsafe_view(narrowed, desiredShape.toArray)
          else narrowed
        if (narrowed != p) {
          narrowed.release
        }
        if (viewed == p) None
        else Some(viewed)
      }
    }

  def toMat(t: Tensor) = {
    if (t.scalarTypeByte() == 6) toFloatMat(t).map(_.toDouble)
    else {
      assert(
        t.scalarTypeByte == 7,
        s"Expected Double Tensor. Got scalartype: ${t.scalarTypeByte}"
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
    if (t.scalarTypeByte() == 6) {
      assert(
        t.numel <= Int.MaxValue,
        "Tensor too long to fit into a java array"
      )
      val arr = Array.ofDim[Float](t.numel.toInt)
      assert(t.copyToFloatArray(arr))
      arr.toVec.map(_.toDouble)
    } else {
      assert(
        t.scalarTypeByte == 7,
        s"Expected Double Tensor. Got scalartype: ${t.scalarTypeByte}"
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
      t.scalarTypeByte == 4,
      s"Expected Long Tensor. Got scalartype: ${t.scalarTypeByte}"
    )
    assert(t.numel <= Int.MaxValue, "Tensor too long to fit into a java array")
    val arr = Array.ofDim[Long](t.numel.toInt)
    assert(t.copyToLongArray(arr))
    arr.toVec
  }
  def toFloatVec(t: Tensor) = {
    assert(
      t.scalarTypeByte == 6,
      s"Expected Float Tensor. Got scalartype: ${t.scalarTypeByte}"
    )
    assert(t.numel <= Int.MaxValue, "Tensor too long to fit into a java array")
    val arr = Array.ofDim[Float](t.numel.toInt)
    assert(t.copyToFloatArray(arr))
    arr.toVec
  }
  def toFloatMat(t: Tensor) = {
    assert(
      t.scalarTypeByte == 6,
      s"Expected Double Tensor. Got scalartype: ${t.scalarTypeByte}"
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
      t.scalarTypeByte == 4,
      s"Expected Long Tensor. Got scalartype: ${t.scalarTypeByte}"
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
      STenOptions.d.value
    )
    assert(t.copyFromDoubleArray(arr))
    if (device != CPU || precision != DoublePrecision) {
      val t2 = Scope.leak { implicit scope =>
        t.to(device.options(precision).value, true, true)
      }
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
      STenOptions.f.value
    )
    assert(t.copyFromFloatArray(arr))
    if (device != CPU) {
      val t2 = Scope.leak { implicit scope =>
        t.to(device.options(SinglePrecision).value, true, true)
      }
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
      STenOptions.d.value
    )
    assert(t.copyFromDoubleArray(arr))
    if (device != CPU || precision != DoublePrecision) {
      val t2 = Scope.leak { implicit scope =>
        t.to(device.options(precision).value, true, true)
      }
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
      STenOptions.l.value
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
      STenOptions.l.value
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
      STenOptions.d.value
    )
    assert(t.copyFromDoubleArray(arr))
    if (device != CPU || precision != DoublePrecision) {
      val t2 = Scope.leak { implicit scope =>
        t.to(device.options(precision).value, true, true)
      }
      t.release
      t2
    } else t
  }
  def fromDoubleArray(
      arr: Array[Double],
      dim: Seq[Long],
      device: Device,
      precision: FloatingPointPrecision
  ) = {
    require(
      arr.length == dim.foldLeft(1L)(_ * _).toInt,
      s"incorrect dimensions $dim got ${arr.length} elements."
    )
    val t = ATen.zeros(
      dim.toArray,
      STenOptions.d.value
    )
    assert(t.copyFromDoubleArray(arr))
    if (device != CPU || precision != DoublePrecision) {
      val t2 = Scope.leak { implicit scope =>
        t.to(device.options(precision).value, true, true)
      }
      t.release
      t2
    } else t
  }
  def fromFloatArray(
      arr: Array[Float],
      dim: Seq[Long],
      device: Device
  ) = {
    require(
      arr.length == dim.foldLeft(1L)(_ * _).toInt,
      s"incorrect dimensions $dim got ${arr.length} elements."
    )
    val t = ATen.zeros(
      dim.toArray,
      STenOptions.f.value
    )
    assert(t.copyFromFloatArray(arr))
    if (device != CPU) {
      val t2 = device.to(t)
      t.release
      t2
    } else t
  }
  def fromLongArray(
      arr: Array[Long],
      dim: Seq[Long],
      device: Device
  ) = {
    require(
      arr.length == dim.foldLeft(1L)(_ * _).toInt,
      s"incorrect dimensions $dim got ${arr.length} elements."
    )
    val t = ATen.zeros(
      dim.toArray,
      STenOptions.l.value
    )
    assert(t.copyFromLongArray(arr))
    if (device != CPU) {
      val t2 = device.to(t)
      t.release
      t2
    } else t
  }

  def device(t: Tensor) = {
    val op = t.options
    if (op.isCPU) CPU
    else CudaDevice(op.deviceIndex.toShort)
  }

  def precision(t: Tensor) = {
    val op = t.options
    if (op.isFloat()) Some(SinglePrecision)
    else if (op.isDouble) Some(DoublePrecision)
    else None
  }

}
