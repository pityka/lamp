package lamp.saddle

import aten.Tensor
import org.saddle._
import aten.ATen
import lamp._

private[lamp] object SaddleTensorHelpers {

  def toMat(t0: Tensor): Mat[Double] = {
    val t = if (t0.isCuda) t0.cpu else t0
    try {
      if (t.scalarTypeByte() != 7) {
        val tmp = ATen._cast_Double(t0, true)
        val mat = toMat(tmp)
        tmp.release
        mat
      } else {
        assert(
          t.scalarTypeByte == 7,
          s"Expected Double Tensor. Got scalartype: ${t.scalarTypeByte}"
        )
        val shape = t.sizes()
        if (shape.size == 2) {
          val arr = Array.ofDim[Double]((shape(0) * shape(1)).toInt)
          require(t.copyToDoubleArray(arr), "failed to copy")
          Mat.apply(shape(0).toInt, shape(1).toInt, arr)
        } else if (shape.size == 0) {
          val arr = Array.ofDim[Double](1)
          require(t.copyToDoubleArray(arr))
          Mat.apply(1, 1, arr)
        } else if (shape.size == 1) {
          val arr = Array.ofDim[Double](shape(0).toInt)
          require(t.copyToDoubleArray(arr))
          Mat.apply(1, shape(0).toInt, arr)
        } else throw new RuntimeException("shape: " + shape.toVector)

      }
    } finally {
      if (t != t0) { t.release }
    }
  }
  def toFloatMat(t0: Tensor) = {
    val t = if (t0.isCuda) t0.cpu else t0
    try {
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
      } else throw new RuntimeException("shape: " + shape.toVector)
    } finally {
      if (t != t0) { t.release }
    }
  }
  def toLongMat(t0: Tensor) = {
    val t = if (t0.isCuda) t0.cpu else t0
    try {
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
    } finally {
      if (t != t0) { t.release }
    }
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

    if (arr.nonEmpty) { assert(t.copyFromDoubleArray(arr)) }

    if (device != CPU || precision != DoublePrecision) {
      val t2 = Scope.unsafe { implicit scope =>
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
    if (arr.nonEmpty) {
      assert(t.copyFromFloatArray(arr))
    }
    if (device != CPU) {
      val t2 = Scope.unsafe { implicit scope =>
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
    if (arr.nonEmpty) {
      assert(t.copyFromDoubleArray(arr))
    }
    if (device != CPU || precision != DoublePrecision) {
      val t2 = Scope.unsafe { implicit scope =>
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
    if (arr.nonEmpty) {
      assert(t.copyFromLongArray(arr))
    }
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
    if (arr.nonEmpty) {
      assert(t.copyFromLongArray(arr))
    }
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
    if (arr.nonEmpty) {
      assert(t.copyFromDoubleArray(arr))
    }
    if (device != CPU || precision != DoublePrecision) {
      val t2 = Scope.unsafe { implicit scope =>
        t.to(device.options(precision).value, true, true)
      }
      t.release
      t2
    } else t
  }

}
