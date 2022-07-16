package lamp

import aten.Tensor
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
      val t2 = Scope.unsafe { implicit scope =>
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
      device: Device
  ) : Tensor = fromLongArray(arr,List(arr.length),device)
  
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
    val r =
      if (op.isCPU) CPU
      else CudaDevice(op.deviceIndex.toShort)
    op.release
    r
  }

  def precision(t: Tensor) = {
    val op = t.options
    val r =
      if (op.isFloat()) Some(SinglePrecision)
      else if (op.isDouble) Some(DoublePrecision)
      else None
    op.release
    r
  }

  def toDoubleArray(t0: Tensor) = {
    val t = if (t0.isCuda) t0.cpu else t0
    try {
      if (t.scalarTypeByte() == 6) {
        assert(
          t.numel <= Int.MaxValue,
          "Tensor too long to fit into a java array"
        )
        val arr = Array.ofDim[Float](t.numel.toInt)
        assert(t.copyToFloatArray(arr))
        arr.map(_.toDouble)
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
        arr
      }
    } finally {
      if (t != t0) { t.release }
    }

  }

  def toLongArray(t0: Tensor) = {
    val t = if (t0.isCuda) t0.cpu else t0
    try {
      assert(
        t.scalarTypeByte == 4,
        s"Expected Long Tensor. Got scalartype: ${t.scalarTypeByte}"
      )
      assert(
        t.numel <= Int.MaxValue,
        "Tensor too long to fit into a java array"
      )
      val arr = Array.ofDim[Long](t.numel.toInt)
      assert(t.copyToLongArray(arr))
      arr
    } finally {
      if (t != t0) { t.release }
    }
  }
  def toFloatArray(t0: Tensor) = {
    val t = if (t0.isCuda) t0.cpu else t0
    try {
      assert(
        t.scalarTypeByte == 6,
        s"Expected Float Tensor. Got scalartype: ${t.scalarTypeByte}"
      )
      assert(
        t.numel <= Int.MaxValue,
        "Tensor too long to fit into a java array"
      )
      val arr = Array.ofDim[Float](t.numel.toInt)
      assert(t.copyToFloatArray(arr))
      arr
    } finally {
      if (t != t0) { t.release }
    }
  }

}
