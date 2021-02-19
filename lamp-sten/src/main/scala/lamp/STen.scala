package lamp

import aten.Tensor
import aten.ATen
import org.saddle._

/** Companion object of [[lamp.STen]]
  *
  * - [[STen.fromDoubleArray]], [[STen.fromLongArray]], [[STen.fromFloatArray]] factory methods
  * copy data from JVM arrays into off heap memory and create an STen instance
  *  - There are similar factories which take SADDLE data structures
  */
object STen {

  /** A tensor option specifying CPU and double */
  val dOptions = STenOptions(aten.TensorOptions.d)

  /** A tensor option specifying CPU and float */
  val fOptions = STenOptions(aten.TensorOptions.f)

  /** A tensor option specifying CPU and long */
  val lOptions = STenOptions(aten.TensorOptions.l)

  implicit class OwnedSyntax(t: Tensor) {
    def owned[S: Sc] = STen.owned(t)
  }

  /** Returns a tensor with the given content and shape on the given device */
  def fromMat[S: Sc](
      m: Mat[Double],
      cuda: Boolean = false
  ) = owned(TensorHelpers.fromMat(m, cuda))

  /** Returns a tensor with the given content and shape on the given device */
  def fromMat[S: Sc](
      m: Mat[Double],
      device: Device,
      precision: FloatingPointPrecision
  ) = owned(TensorHelpers.fromMat(m, device, precision))

  /** Returns a tensor with the given content and shape on the given device */
  def fromFloatMat[S: Sc](
      m: Mat[Float],
      device: Device
  ) = owned(TensorHelpers.fromFloatMat(m, device))

  /** Returns a tensor with the given content and shape on the given device */
  def fromVec[S: Sc](
      m: Vec[Double],
      cuda: Boolean = false
  ) = owned(TensorHelpers.fromVec(m, cuda))

  /** Returns a tensor with the given content and shape on the given device */
  def fromVec[S: Sc](
      m: Vec[Double],
      device: Device,
      precision: FloatingPointPrecision
  ) = owned(TensorHelpers.fromVec(m, device, precision))

  /** Returns a tensor with the given content and shape on the given device */
  def fromLongMat[S: Sc](
      m: Mat[Long],
      device: Device
  ) = owned(TensorHelpers.fromLongMat(m, device))

  /** Returns a tensor with the given content and shape on the given device */
  def fromLongMat[S: Sc](
      m: Mat[Long],
      cuda: Boolean = false
  ) = owned(TensorHelpers.fromLongMat(m, cuda))

  /** Returns a tensor with the given content and shape on the given device */
  def fromLongVec[S: Sc](
      m: Vec[Long],
      device: Device
  ) = owned(TensorHelpers.fromLongVec(m, device))

  /** Returns a tensor with the given content and shape on the given device */
  def fromLongVec[S: Sc](
      m: Vec[Long],
      cuda: Boolean = false
  ) = owned(TensorHelpers.fromLongVec(m, cuda))

  /** Returns a tensor with the given content and shape on the given device */
  def fromLongArray[S: Sc](ar: Array[Long], dim: Seq[Long], device: Device) =
    TensorHelpers.fromLongArray(ar, dim, device).owned

  /** Returns a tensor with the given content and shape on the given device */
  def fromDoubleArray[S: Sc](
      ar: Array[Double],
      dim: Seq[Long],
      device: Device,
      precision: FloatingPointPrecision
  ) =
    TensorHelpers.fromDoubleArray(ar, dim, device, precision).owned

  /** Returns a tensor with the given content and shape on the given device */
  def fromFloatArray[S: Sc](ar: Array[Float], dim: Seq[Long], device: Device) =
    TensorHelpers.fromFloatArray(ar, dim, device).owned

  /** Wraps a tensor without registering it to any scope.
    *
    * Memory may leak.
    */
  def free(value: Tensor) = STen(value)

  /** Returns a 1D tensor containing the given values */
  def apply[S: Sc](vs: Double*) = fromVec(Vec(vs: _*))

  /** Wraps an aten.Tensor and registering it to the given scope */
  def owned(
      value: Tensor
  )(implicit scope: Scope): STen = {
    scope(value)
    STen(value)
  }

  def cat[S: Sc](tensors: Seq[STen], dim: Long) =
    owned(ATen.cat(tensors.map(_.value).toArray, dim))
  def stack[S: Sc](tensors: Seq[STen], dim: Long) =
    owned(ATen.stack(tensors.map(_.value).toArray, dim))

  def scalarLong(value: Long, options: STenOptions)(implicit scope: Scope) =
    Tensor.scalarLong(value, options.toLong.value).owned
  def scalarDouble[S: Sc](value: Double, options: STenOptions) =
    Tensor.scalarDouble(value, options.toDouble.value).owned

  def ones[S: Sc](
      size: Seq[Long],
      tensorOptions: STenOptions = STen.dOptions
  ) =
    owned(ATen.ones(size.toArray.map(_.toLong), tensorOptions.value))

  def onesLike[S: Sc](
      tensor: Tensor
  ) =
    owned(Tensor.ones_like(tensor))
  def onesLike[S: Sc](
      tensor: STen
  ) =
    owned(Tensor.ones_like(tensor.value))

  def zeros[S: Sc](
      size: Seq[Long],
      tensorOptions: STenOptions = STen.dOptions
  ) =
    owned(ATen.zeros(size.toArray.map(_.toLong), tensorOptions.value))
  def zerosLike[S: Sc](
      tensor: Tensor
  ) =
    owned(Tensor.zeros_like(tensor))
  def zerosLike[S: Sc](
      tensor: STen
  ) =
    owned(Tensor.zeros_like(tensor.value))
  def rand[S: Sc](
      size: Seq[Long],
      tensorOptions: STenOptions = STen.dOptions
  ) =
    owned(ATen.rand(size.toArray.map(_.toLong), tensorOptions.value))
  def randn[S: Sc](
      size: Seq[Long],
      tensorOptions: STenOptions = STen.dOptions
  ) =
    owned(ATen.randn(size.toArray.map(_.toLong), tensorOptions.value))

  def normal[S: Sc](
      mean: Double,
      std: Double,
      size: Seq[Long],
      options: STenOptions
  ) =
    ATen.normal_3(mean, std, size.toArray, options.value).owned

  def randperm[S: Sc](
      n: Long,
      tensorOptions: STenOptions = STen.dOptions
  ) =
    owned(
      ATen.randperm_0(n, tensorOptions.value)
    )

  def randint[S: Sc](
      high: Long,
      size: Seq[Long],
      tensorOptions: STenOptions = STen.dOptions
  ) =
    owned(
      ATen.randint_0(high, size.toArray, tensorOptions.value)
    )
  def randint[S: Sc](
      low: Long,
      high: Long,
      size: Seq[Long],
      tensorOptions: STenOptions
  ) =
    owned(
      ATen.randint_2(low, high, size.toArray, tensorOptions.value)
    )
  def eye[S: Sc](n: Int, tensorOptions: STenOptions = STen.dOptions) =
    owned(ATen.eye_0(n, tensorOptions.value))
  def eye[S: Sc](
      n: Int,
      m: Int,
      tensorOptions: STenOptions
  ) =
    owned(ATen.eye_1(n, m, tensorOptions.value))

  def arange[S: Sc](
      start: Double,
      end: Double,
      step: Double,
      tensorOptions: STenOptions = STen.dOptions
  ) =
    owned(ATen.arange(start, end, step, tensorOptions.value))
  def linspace[S: Sc](
      start: Double,
      end: Double,
      steps: Long,
      tensorOptions: STenOptions = STen.dOptions
  ) =
    owned(ATen.linspace(start, end, steps, tensorOptions.value))
  def sparse_coo[S: Sc](
      indices: STen,
      values: STen,
      dim: Seq[Long],
      tensorOptions: STenOptions = STen.dOptions
  ) =
    owned(
      ATen.sparse_coo_tensor(
        indices.value,
        values.value,
        dim.toArray,
        tensorOptions.value
      )
    )

  def indexSelectOut(out: STen, self: STen, dim: Int, index: STen) =
    ATen.index_select_out(out.value, self.value, dim, index.value)
  def catOut(out: STen, tensors: Seq[STen], dim: Int) =
    ATen.cat_out(out.value, tensors.map(_.value).toArray, dim)
  def addOut(
      out: STen,
      self: STen,
      other: STen,
      alpha: Double
  ): Unit =
    ATen.add_out(out.value, self.value, other.value, alpha)
  def subOut(
      out: STen,
      self: STen,
      other: STen,
      alpha: Double
  ): Unit =
    ATen.sub_out(out.value, self.value, other.value, alpha)
  def mulOut(
      out: STen,
      self: STen,
      other: STen
  ): Unit =
    ATen.mul_out(out.value, self.value, other.value)
  def divOut(
      out: STen,
      self: STen,
      other: STen
  ): Unit =
    ATen.div_out(out.value, self.value, other.value)
  def mmOut(
      out: STen,
      self: STen,
      other: STen
  ): Unit =
    ATen.mm_out(out.value, self.value, other.value)
  def bmmOut(
      out: STen,
      self: STen,
      other: STen
  ): Unit =
    ATen.bmm_out(out.value, self.value, other.value)
  def remainderOut(
      out: STen,
      self: STen,
      other: STen
  ): Unit =
    ATen.remainder_out_1(out.value, self.value, other.value)
  def remainderOut(
      out: STen,
      self: STen,
      other: Double
  ): Unit =
    ATen.remainder_out_0(out.value, self.value, other)
  def powOut(
      out: STen,
      self: STen,
      other: Double
  ): Unit =
    ATen.pow_out_0(out.value, self.value, other)
  def powOut(
      out: STen,
      self: STen,
      other: STen
  ): Unit =
    ATen.pow_out_1(out.value, self.value, other.value)
  def sumOut(
      out: STen,
      self: STen,
      dim: Seq[Int],
      keepDim: Boolean
  ): Unit =
    ATen.sum_out(out.value, self.value, dim.map(_.toLong).toArray, keepDim)
  def meanOut(
      out: STen,
      self: STen,
      dim: Seq[Int],
      keepDim: Boolean
  ): Unit =
    ATen.mean_out(out.value, self.value, dim.map(_.toLong).toArray, keepDim)
  def addmmOut(
      out: STen,
      self: STen,
      mat1: STen,
      mat2: STen,
      beta: Double,
      alpha: Double
  ): Unit =
    ATen.addmm_out(out.value, self.value, mat1.value, mat2.value, beta, alpha)

  def addcmulOut(
      out: STen,
      self: STen,
      tensor1: STen,
      tensor2: STen,
      alpha: Double
  ): Unit =
    ATen
      .addcmul_out(out.value, self.value, tensor1.value, tensor2.value, alpha)

  def tanh_backward[S: Sc](gradOutput: STen, output: STen) =
    ATen.tanh_backward(gradOutput.value, output.value).owned
  def l1_loss_backward[S: Sc](
      gradOutput: STen,
      self: STen,
      target: STen,
      reduction: Long
  ) =
    ATen
      .l1_loss_backward(gradOutput.value, self.value, target.value, reduction)
      .owned
  def mse_loss_backward[S: Sc](
      gradOutput: STen,
      self: STen,
      target: STen,
      reduction: Long
  ) =
    ATen
      .mse_loss_backward(gradOutput.value, self.value, target.value, reduction)
      .owned
  def mse_loss[S: Sc](
      self: STen,
      target: STen,
      reduction: Long
  ) =
    ATen.mse_loss(self.value, target.value, reduction).owned

  def where[S: Sc](condition: STen, self: STen, other: STen) =
    ATen.where_0(condition.value, self.value, other.value).owned
  def where[S: Sc](condition: Tensor, self: STen, other: STen) =
    ATen.where_0(condition, self.value, other.value).owned

}

case class STenOptions(value: aten.TensorOptions) {
  import STenOptions._

  /** Returns a copy with dtype set to long */
  def toLong[S: Sc] = value.toLong.owned

  /** Returns a copy with dtype set to double */
  def toDouble[S: Sc] = value.toDouble.owned

  /** Returns a copy with dtype set to float */
  def toFloat[S: Sc] = value.toFloat.owned

  /** Returns a copy with device set to CPU */
  def cpu[S: Sc] = value.cpu.owned

  /** Returns a copy with device set to cuda with index */
  def cudaIndex[S: Sc](index: Short) = value.cuda_index(index).owned

  /** Returns a copy with device set to cuda:0 */
  def cuda[S: Sc] = cudaIndex(0)

  def isDouble = value.isDouble
  def isFloat = value.isFloat
  def isLong = value.isLong
  def isCPU = value.isCPU
  def isCuda = value.isCuda
  def isSparse = value.isSparse
  def deviceIndex = value.deviceIndex

  /** Returns the byte representation of dtype
    *
    * 4 - long, 6 - float, 7 - oble
    */
  def scalarTypeByte = value.scalarTypeByte
}
object STenOptions {

  /** Returns an tensor option specifying CPU and double */
  def d = STen.dOptions

  /** Returns an tensor option specifying CPU and float */
  def f = STen.fOptions

  /** Returns an tensor option specifying CPU and long */
  def l = STen.lOptions

  /** Returns an tensor option specifying CPU and dtype corresponding to the given byte
    *
    * 4 - long, 6 - float, 7 - double
    */
  def fromScalarType[S: Sc](b: Byte) =
    owned(aten.TensorOptions.fromScalarType(b))
  implicit class OwnedSyntaxOp(t: aten.TensorOptions) {
    def owned[S: Sc] = STenOptions.owned(t)
  }
  def owned(
      value: aten.TensorOptions
  )(implicit scope: Scope): STenOptions = {
    scope(value)
    STenOptions(value)
  }
}

/** Memory managed, off-heap N-dimensional array.
  *
  * This class is a wrapper around aten.Tensor providing a more convenient API.
  * All allocating operations require an implicit [[lamp.Scope]].
  *
  * STen instances are associated with a device which determines where the memory is allocated,
  * and where the operations are performed.
  * Operations on multiple tensors expect that all the arguments reside on the same device.
  *
  * [[lamp.STen.options]] returns a [[lamp.STenOptions]] which describes the device, shape, data type and
  * storage layout of a tensor.
  * Most factory methods in the companion object in turn require a [[lamp.STenOptions]] to specify
  * the device, data types and storage layout.
  *
  * Naming convention of most operations follows libtorch.
  * Operations return their result in a copy, i.e. not in place. These operations need a [[lamp.Scope]].
  * Operations whose name ends with an underscore are in place.
  * Operations whose name contains `out` will write their results into the specified output tensor, these are in the companion object.
  * Some operations are exempt from this naming rule, e.g. `+=`, `-=`, `*=` etc.
  *
  * Semantics of operations follow those of libtorch with the same name.
  * Many of the operations broadcasts. See [[https://numpy.org/doc/stable/user/basics.broadcasting.html#general-broadcasting-rules]] for broadcasting rules. In short:
  *
  *  1. shapes are aligned from the right, extending with ones to the left as needed.
  *  2. If two aligned dimensions are not matching but one of them is 1, then it is expanded to the
  *     size of the other dimension, pretending a copy of all its values. If two aligned dimension are not matching and neither of them is 1, then the operation fails.
  * =Examples=
  * {{{
  * Scope.root { implicit scope =>
  *    val sum = Scope { implicit scope =>
  *     val ident = STen.eye(3, STenOptions.d)
  *     val ones = STen.ones(List(3, 3), STenOptions.d)
  *     ident + ones
  *    }
  *    assert(sum.toMat == mat.ones(3, 3) + mat.ident(3))
  * }
  * }}}
  * ===Broadcasting examples===
  * {{{
  * // successful
  * 3 x 4 x 6 A
  *     4 x 6 B
  * 3 x 4 x 6 Result // B is repeated 3 times first dimensions
  *
  * // successful
  * 3 x 4 x 6 A
  * 3 x 1 x 6 B
  * 3 x 4 x 6 Result // B's second dimension is repeated 4 times
  *
  * // fail
  * 3 x 4 x 6 A
  * 3 x 2 x 6 B
  * 3 x 4 x 6 Result // 2 != 4
  * }}}
  *
  * The companion object contains various factories which copy data from the JVM memory to STen tensors.
  *
  *
  */
case class STen private (
    value: Tensor
) {
  import STen._

  /** Returns the number of elements in the tensor */
  def numel = value.numel

  /** Converts to a Mat[Double].
    *
    * Copies to CPU if needed.
    * Fails if dtype is not float or double.
    * Fails if shape does not conform a matrix.
    */
  def toMat = TensorHelpers.toMat(value)

  /** Converts to a Mat[Float].
    *
    * Copies to CPU if needed.
    * Fails if dtype is not float.
    * Fails if shape does not conform a matrix.
    */
  def toFloatMat = TensorHelpers.toFloatMat(value)

  /** Converts to a Mat[Long].
    *
    * Copies to CPU if needed.
    * Fails if dtype is not long.
    * Fails if shape does not conform a matrix.
    */
  def toLongMat = TensorHelpers.toLongMat(value)

  /** Converts to a Vec[Double].
    *
    * Copies to CPU if needed.
    * Fails if dtype is not float or double.
    * Flattens the shape.
    */
  def toVec = TensorHelpers.toVec(value)

  /** Converts to a Vec[Float].
    *
    * Copies to CPU if needed.
    * Fails if dtype is not float.
    * Flattens the shape.
    */
  def toFloatVec = TensorHelpers.toFloatVec(value)

  /** Converts to a Vec[Long].
    *
    * Copies to CPU if needed.
    * Fails if dtype is not long.
    * Flattens the shape.
    */
  def toLongVec = TensorHelpers.toLongVec(value)

  /** Returns the shape of the tensor */
  def shape = value.sizes.toList

  /** Returns the shape of the tensor */
  def sizes = shape

  /** Returns the associated STenOptions */
  def options[S: Sc] = STenOptions.owned(value.options())
  def coalesce[S: Sc] = value.coalesce.owned

  /** Returns indices. Only for sparse tensors */
  def indices[S: Sc] = value.indices.owned

  /** Returns values. Only for sparse tensors */
  def values[S: Sc] = value.indices.owned

  /** Returns true if data type is double */
  def isDouble = Scope.leak { implicit scope => options.isDouble }

  /** Returns true if data type is float */
  def isFloat = Scope.leak { implicit scope => options.isFloat }

  /** Returns true if data type is long */
  def isLong = Scope.leak { implicit scope => options.isLong }

  /** Returns true if device is CPU */
  def isCPU = Scope.leak { implicit scope => options.isCPU }

  /** Returns true if device is Cuda */
  def isCuda = Scope.leak { implicit scope => options.isCuda }

  /** Returns true if this is sparse tensor */
  def isSparse = Scope.leak { implicit scope => options.isSparse }

  /** Returns the device index. Only for Cuda tensors. */
  def deviceIndex = Scope.leak { implicit scope => options.deviceIndex }

  /** Returns the byte representation of the data type
    *
    * The mapping is:
    *  - 4 for Long
    *  - 6 for Float
    *  - 7 for Double
    */
  def scalarTypeByte = Scope.leak { implicit scope => options.scalarTypeByte }

  /** Returns a copy of this tensor on the given device */
  def copyToDevice(device: Device)(implicit scope: Scope) = {
    STen.owned(device.to(value))
  }

  /** Returns a copy of this tensor */
  def cloneTensor[S: Sc] = ATen.clone(value).owned

  /** Returns a copy of this tensor adapted to the given options */
  def copyTo[S: Sc](options: STenOptions) =
    value.to(options.value, true, true).owned

  /** Overwrites the contents of this tensor with the contents of an other. Must conform. */
  def copyFrom(source: Tensor) =
    value.copyFrom(source)

  /** Overwrites the contents of this tensor with the contents of an other. Must conform. */
  def copyFrom(source: STen) =
    value.copyFrom(source.value)

  override def toString =
    s"STen(shape=$shape,value=$value)"

  def unbroadcast[S: Sc](sizes: Seq[Long]) =
    TensorHelpers.unbroadcast(value, sizes.toList) match {
      case None    => this
      case Some(t) => t.owned
    }

  /** Transposes the first two dimensions. */
  def t[S: Sc] = owned(ATen.t(value))

  /** Transposes the  given dimensions. */
  def transpose[S: Sc](dim1: Int, dim2: Int) =
    owned(ATen.transpose(value, dim1, dim2))

  /** Selects a scalar element or a tensor in the given dimension and index. */
  def select[S: Sc](dim: Long, index: Long) =
    owned(ATen.select(value, dim, index))

  /** Flattens between the given dimensions. Inclusive. */
  def flatten[S: Sc](startDim: Long, endDim: Long) =
    owned(ATen.flatten(value, startDim, endDim))

  /** Selects along the given dimension with indices in the supplied long tensor. */
  def indexSelect[S: Sc](dim: Long, index: STen) =
    owned(ATen.index_select(value, dim, index.value))

  /** Selects along the given dimension with indices in the supplied long tensor. */
  def indexSelect[S: Sc](dim: Long, index: Tensor) =
    owned(ATen.index_select(value, dim, index))

  /** Reduces the given dimension with the index of its maximum element.
    *
    * @param keepDim if true then the reduced dimension is kept with size 1
    */
  def argmax[S: Sc](dim: Long, keepDim: Boolean) =
    owned(ATen.argmax(value, dim, keepDim))

  /** Reduces the given dimension with the index of its minimum element.
    *
    * @param keepDim if true then the reduced dimension is kept with size 1
    */
  def argmin[S: Sc](dim: Long, keepDim: Boolean) =
    owned(ATen.argmin(value, dim, keepDim))

  /** Fills the tensor with the given `fill` value in the locations indicated by the `mask` boolean mask. */
  def maskFill[S: Sc](mask: STen, fill: Double) =
    owned(ATen.masked_fill_0(value, mask.value, fill))

  /** Returns a boolean tensors of the same shape, indicating equality with the other tensor. */
  def equ[S: Sc](other: STen) =
    owned(ATen.eq_1(value, other.value))

  /** Returns a boolean tensors of the same shape, indicating equality with the other value. */
  def equ[S: Sc](other: Double) =
    owned(ATen.eq_0(value, other))

  /** Returns a boolean tensors of the same shape, indicating equality with the other value. */
  def equ[S: Sc](other: Long) =
    owned(ATen.eq_0_l(value, other))

  /** Concatenates two tensors along the given dimension. Other dimensions must conform. */
  def cat[S: Sc](other: STen, dim: Long) =
    owned(ATen.cat(Array(value, other.value), dim))

  def unique[S: Sc](sorted: Boolean, returnInverse: Boolean) = {
    val (a, b) =
      ATen._unique(value, sorted, returnInverse)
    (owned(a), owned(b))
  }

  def unique[S: Sc](
      sorted: Boolean,
      returnInverse: Boolean,
      returnCounts: Boolean
  ) = {
    val (a, b, c) =
      ATen._unique2(value, sorted, returnInverse, returnCounts)
    (owned(a), owned(b), owned(c))
  }
  def unique[S: Sc](
      dim: Int,
      sorted: Boolean,
      returnInverse: Boolean,
      returnCounts: Boolean
  ) = {
    val (a, b, c) =
      ATen.unique_dim(value, dim, sorted, returnInverse, returnCounts)
    (owned(a), owned(b), owned(c))
  }
  def uniqueConsecutive[S: Sc](
      dim: Int,
      returnInverse: Boolean = false,
      returnCounts: Boolean = false
  ) = {
    val (a, b, c) =
      ATen.unique_consecutive(value, returnInverse, returnCounts, dim)
    (owned(a), owned(b), owned(c))
  }

  /** Casts to char */
  def castToChar[S: Sc] = owned(ATen._cast_Char(value, true))

  /** Casts to byte */
  def castToByte[S: Sc] = owned(ATen._cast_Byte(value, true))

  /** Casts to float */
  def castToFloat[S: Sc] = owned(ATen._cast_Float(value, true))

  /** Casts to double */
  def castToDouble[S: Sc] = owned(ATen._cast_Double(value, true))

  /** Casts to long */
  def castToLong[S: Sc] = owned(ATen._cast_Long(value, true))

  /** Adds to tensors. */
  def +[S: Sc](other: STen) =
    owned(ATen.add_0(value, other.value, 1d))

  /** In place add. */
  def +=(other: STen): Unit =
    ATen.add_out(value, value, other.value, 1d)

  /** In place add. */
  def +=(other: Double): Unit =
    value.add_(other, 1d)

  /** Adds a scalar to all elements. */
  def +[S: Sc](other: Double) =
    owned(ATen.add_1(value, other, 1d))

  /** Adds an other tensor multipled by a scalar `(a + alpha * b)`. */
  def add[S: Sc](other: STen, alpha: Double) =
    owned(ATen.add_0(value, other.value, alpha))

  /** Adds a value multipled by a scalar `(a + alpha * b)`. */
  def add[S: Sc](other: Double, alpha: Double) =
    owned(ATen.add_1(value, other, alpha))

  /** `beta * this + alpha * (mat1 matmul mat2)` */
  def addmm[S: Sc](mat1: STen, mat2: STen, beta: Double, alpha: Double) =
    ATen.addmm(value, mat1.value, mat2.value, beta, alpha).owned

  /** Subtracts other. */
  def -[S: Sc](other: STen) =
    owned(ATen.sub_0(value, other.value, 1d))

  /** Subtracts other in place. */
  def -=[S: Sc](other: STen): Unit =
    ATen.sub_out(value, value, other.value, 1d)

  /** Subtracts other after multiplying with a number. */
  def sub[S: Sc](other: STen, alpha: Double) =
    owned(ATen.sub_0(value, other.value, alpha))

  /** Subtracts other after multiplying with a number. */
  def sub[S: Sc](other: Double, alpha: Double) =
    owned(ATen.sub_1(value, other, alpha))

  /** Multiplication */
  def *[S: Sc](other: STen) =
    owned(ATen.mul_0(value, other.value))

  /** Multiplication */
  def *[S: Sc](other: Tensor) =
    owned(ATen.mul_0(value, other))

  /** Multiplication */
  def *[S: Sc](other: Double) =
    owned(ATen.mul_1(value, other))

  /** In place multiplication. */
  def *=[S: Sc](other: STen): Unit =
    ATen.mul_out(value, value, other.value)

  /** In place multiplication. */
  def *=[S: Sc](other: Double): Unit =
    value.mul_(other)

  /** Division. */
  def /[S: Sc](other: STen) =
    owned(ATen.div_0(value, other.value))

  /** Division. */
  def /[S: Sc](other: Tensor) =
    owned(ATen.div_0(value, other))

  /** Division. */
  def /[S: Sc](other: Double) =
    owned(ATen.div_1(value, other))

  /** In place division. */
  def /=[S: Sc](other: STen): Unit =
    ATen.div_out(value, value, other.value)

  /** Matrix multiplication. Maps to Aten.mm. */
  def mm[S: Sc](other: STen) =
    owned(ATen.mm(value, other.value))

  def matmul[S: Sc](other: STen) =
    owned(ATen.matmul(value, other.value))

  def dot[S: Sc](other: STen) =
    owned(ATen.dot(value, other.value))

  /** Batched matrix multiplication. Maps to Aten.bmm.
    *
    * Performs the same matrix multiplication along multiple batches.
    * Batch dimensions do not broadcast.
    */
  def bmm[S: Sc](other: STen) =
    owned(ATen.bmm(value, other.value))

  /** Batched add mm. */
  def baddbmm[S: Sc](batch1: STen, batch2: STen, beta: Double, alpha: Double) =
    ATen.baddbmm(value, batch1.value, batch2.value, beta, alpha).owned

  /** Elementwise `this + alpha * tensor1 * tensor2` */
  def addcmul[S: Sc](
      tensor1: STen,
      tensor2: STen,
      alpha: Double
  ) =
    ATen.addcmul(value, tensor1.value, tensor2.value, alpha).owned

  /** Elementwise in place `this + alpha * tensor1 * tensor2` */
  def addcmulSelf(
      tensor1: STen,
      tensor2: STen,
      alpha: Double
  ): Unit =
    ATen.addcmul_out(value, value, tensor1.value, tensor2.value, alpha)

  /** Elementwise in place `this + alpha * tensor1 * tensor2` */
  def addcmulSelf(
      tensor1: STen,
      tensor2: Tensor,
      alpha: Double
  ): Unit =
    ATen.addcmul_out(value, value, tensor1.value, tensor2, alpha)

  /** Rectified linear unit */
  def relu[S: Sc] = owned(ATen.relu(value))

  /** In place rectified linear unit */
  def relu_() = ATen.relu_(value)

  /** Leaky rectified linear unit */
  def leakyRelu[S: Sc](negativeSlope: Double) =
    owned(ATen.leaky_relu(value, negativeSlope))

  /** In place leaky rectified linear unit */
  def leakyRelu_(negativeSlope: Double) = ATen.leaky_relu_(value, negativeSlope)

  /** Gaussian Error Linear Unit */
  def gelu[S: Sc] = owned(ATen.gelu(value))

  /** Sigmoid funtion */
  def sigmoid[S: Sc] = owned(ATen.sigmoid(value))

  /** In place sigmoid funtion */
  def sigmoid_() = ATen.sigmoid_(value)
  def sign[S: Sc] = owned(ATen.sign(value))

  def sign_() = ATen.sign_out(value, value)
  def exp[S: Sc] = owned(ATen.exp(value))
  def exp_() = ATen.exp_(value)
  def log[S: Sc] = owned(ATen.log(value))
  def log_() = ATen.log_(value)
  def log1p[S: Sc] = owned(ATen.log1p(value))
  def log1p_() = ATen.log1p_(value)
  def sin[S: Sc] = owned(ATen.sin(value))
  def sin_() = ATen.sin_(value)
  def cos[S: Sc] = owned(ATen.cos(value))
  def cos_() = ATen.cos_(value)
  def tan[S: Sc] = owned(ATen.tan(value))
  def tan_() = ATen.tan_(value)
  def tanh[S: Sc] = owned(ATen.tanh(value))
  def tanh_() = ATen.tanh_(value)
  def atan[S: Sc] = owned(ATen.atan(value))
  def atan_() = ATen.atan_(value)
  def acos[S: Sc] = owned(ATen.acos(value))
  def acos_() = ATen.acos_(value)
  def asin[S: Sc] = owned(ATen.asin(value))
  def asin_() = ATen.asin_(value)
  def sqrt[S: Sc] = owned(ATen.sqrt(value))
  def sqrt_[S: Sc] = ATen.sqrt_(value)
  def square[S: Sc] = owned(ATen.square(value))
  def square_() = ATen.square_(value)
  def abs[S: Sc] = owned(ATen.abs(value))
  def abs_() = ATen.abs_(value)
  def ceil[S: Sc] = owned(ATen.ceil(value))
  def ceil_() = ATen.ceil_(value)
  def floor[S: Sc] = owned(ATen.floor(value))
  def floor_() = ATen.floor_(value)
  def reciprocal[S: Sc] = owned(ATen.reciprocal(value))
  def reciprocal_() = ATen.reciprocal_(value)
  def det[S: Sc] = owned(ATen.det(value))
  def trace[S: Sc] = owned(ATen.trace(value))

  def remainder[S: Sc](other: STen) =
    ATen.remainder_1(value, other.value).owned

  def remainder[S: Sc](other: Double) =
    ATen.remainder_0(value, other).owned

  def pow[S: Sc](exponent: Double) = owned(ATen.pow_0(value, exponent))
  def pow[S: Sc](exponent: STen) =
    owned(ATen.pow_1(value, exponent.value))
  def pow_(exponent: Double) =
    ATen.pow_out_0(value, value, exponent)

  def sum[S: Sc] = owned(ATen.sum_0(value))

  /** Reduces the given dimensions with the sum of their elements.
    *
    * @param keepDim if true then the reduced dimensions are kept with size 1
    */
  def sum[S: Sc](dim: Seq[Int], keepDim: Boolean) =
    owned(ATen.sum_1(value, dim.toArray.map(_.toLong), keepDim))
  def sum[S: Sc](dim: Int, keepDim: Boolean) =
    owned(ATen.sum_1(value, Array(dim.toLong), keepDim))

  /** Sum over the second dimension */
  def rowSum[S: Sc] = sum(1, true)

  /** Sum over the first dimension */
  def colSum[S: Sc] = sum(0, true)

  /** Selects the top k elements along the given dimension
    *
    * @param k How many elements to select
    * @param dim which dimension to select in
    * @param largest if true, then the highest k element is selected
    * @param sorted if true, the selected elements are further sorted
    * @return a pair of (value,index) tensors where value holds the selected elements and
    * index holds the indices of the selected elements
    */
  def topk[S: Sc](k: Int, dim: Int, largest: Boolean, sorted: Boolean) = {
    val (a, b) = ATen.topk(value, k, dim, largest, sorted)
    (owned(a), owned(b))
  }

  /** Returns a slice over the selected dimension */
  def slice[S: Sc](dim: Int, start: Long, end: Long, step: Long) =
    owned(ATen.slice(value, dim, start, end, step))

  /** Returns a slice over the selected dimension */
  def slice[S: Sc](dim: Long, start: Long, end: Long, step: Long) =
    owned(ATen.slice(value, dim, start, end, step))

  /** In place fills the tensors with the given value */
  def fill_(v: Double) = ATen.fill__0(value, v)

  /** Selects the elements according to the boolean mask. Returns a 1D tensor. */
  def maskedSelect[S: Sc](mask: STen) =
    ATen.masked_select(value, mask.value).owned

  /** Selects the elements according to the boolean mask. Returns a 1D tensor. */
  def maskedSelect[S: Sc](mask: Tensor) =
    ATen.masked_select(value, mask).owned

  /** Fills with the given value according to the boolean mask. */
  def maskedFill[S: Sc](mask: STen, fill: Double) =
    ATen.masked_fill_0(value, mask.value, fill).owned

  /** Fills with the given value according to the boolean mask. */
  def maskedFill[S: Sc](mask: Tensor, fill: Double) =
    ATen.masked_fill_0(value, mask, fill).owned

  def maskedScatter[S: Sc](mask: STen, src: STen) =
    ATen.masked_scatter(value, mask.value, src.value).owned

  /** In place fills with zeros. */
  def zero_(): Unit =
    ATen.zero_(value)

  /** In place fills with the given tensor. */
  def fill_(v: STen) = ATen.fill__1(value, v.value)

  def scatter[S: Sc](
      dim: Long,
      index: STen,
      source: STen
  ) =
    owned(ATen.scatter_0(value, dim, index.value, source.value))
  def scatter[S: Sc](
      dim: Long,
      index: STen,
      source: Double
  ) =
    owned(ATen.scatter_1(value, dim, index.value, source))
  def scatterAdd[S: Sc](
      dim: Long,
      index: STen,
      source: STen
  ) =
    owned(ATen.scatter_add(value, dim, index.value, source.value))

  def indexAdd[S: Sc](
      dim: Long,
      index: STen,
      source: STen
  ) =
    owned(ATen.index_add(value, dim, index.value, source.value))
  def indexAdd[S: Sc](
      dim: Long,
      index: Tensor,
      source: STen
  ) =
    owned(ATen.index_add(value, dim, index, source.value))
  def indexFill[S: Sc](
      dim: Long,
      index: STen,
      source: STen
  ) =
    owned(ATen.index_fill_1(value, dim, index.value, source.value))
  def indexFill[S: Sc](
      dim: Long,
      index: STen,
      source: Double
  ) =
    owned(ATen.index_fill_0(value, dim, index.value, source))
  def indexFill[S: Sc](
      dim: Long,
      index: Tensor,
      source: Double
  ) =
    owned(ATen.index_fill_0(value, dim, index, source))
  def gather[S: Sc](
      dim: Long,
      index: Tensor
  ) =
    owned(ATen.gather(value, dim, index, false))
  def gather[S: Sc](
      dim: Long,
      index: STen
  ) =
    owned(ATen.gather(value, dim, index.value, false))

  def expandAs[S: Sc](other: STen) =
    owned(value.expand_as(other.value))

  /** Returns a tensor with a new shape.
    *
    * No data is copied.
    * The new shape must be compatible with the number of elements and the stride of the tensor.
    */
  def view[S: Sc](dims: Long*) =
    owned(ATen._unsafe_view(value, dims.toArray))

  /** Returns a tensor with a new shape. May copy. */
  def reshape[S: Sc](dims: Long*) =
    owned(ATen.reshape(value, dims.toArray))

  /** Reduces the given dimensions with their L2 norm. */
  def norm2[S: Sc](dim: Seq[Int], keepDim: Boolean) =
    owned(
      ATen.norm_2(
        value,
        dim.toArray.map(_.toLong),
        keepDim,
        STen.dOptions.scalarTypeByte
      )
    )

  /** Reduces the given dimension with the log-softmax of its elements. */
  def logSoftMax[S: Sc](dim: Int) =
    owned(
      ATen.log_softmax(
        value,
        dim
      )
    )

  def mean[S: Sc] =
    owned(ATen.mean_0(value))

  /** Reduces the given dimensions with their mean. */
  def mean[S: Sc](dim: Seq[Int], keepDim: Boolean) =
    owned(ATen.mean_1(value, dim.toArray.map(_.toLong), keepDim))
  def mean[S: Sc](dim: Int, keepDim: Boolean) =
    owned(ATen.mean_1(value, Array(dim), keepDim))

  def variance[S: Sc](unbiased: Boolean) =
    owned(ATen.var_0(value, unbiased))

  /** Reduces the given dimensions with their variance. */
  def variance[S: Sc](dim: Seq[Int], unbiased: Boolean, keepDim: Boolean) =
    owned(ATen.var_1(value, dim.toArray.map(_.toLong), unbiased, keepDim))
  def variance[S: Sc](dim: Int, unbiased: Boolean, keepDim: Boolean) =
    owned(ATen.var_1(value, Array(dim), unbiased, keepDim))

  def std[S: Sc](unbiased: Boolean) =
    owned(ATen.std_0(value, unbiased))

  /** Reduces the given dimensions with their standard deviation. */
  def std[S: Sc](dim: Seq[Int], unbiased: Boolean, keepDim: Boolean) =
    owned(ATen.std_1(value, dim.toArray.map(_.toLong), unbiased, keepDim))
  def std[S: Sc](dim: Int, unbiased: Boolean, keepDim: Boolean) =
    owned(ATen.std_1(value, Array(dim), unbiased, keepDim))

  def median[S: Sc] = owned(ATen.median_1(value))

  /** Reduces the given dimension with its median. */
  def median[S: Sc](dim: Int, keepDim: Boolean) = {
    val (a, b) = ATen.median_0(value, dim, keepDim)
    owned(a) -> owned(b)
  }

  /** Reduces the given dimension with its mode. */
  def mode[S: Sc](dim: Int, keepDim: Boolean) = {
    val (a, b) = ATen.mode(value, dim, keepDim)
    owned(a) -> owned(b)
  }

  def max[S: Sc] = owned(ATen.max_2(value))

  /** Return a boolean tensor indicating elementwise max. */
  def max[S: Sc](other: STen) = owned(ATen.max_1(value, other.value))

  /** Reduces the given dimension with its max. */
  def max[S: Sc](dim: Int, keepDim: Boolean) = {
    val (a, b) = ATen.max_0(value, dim, keepDim)
    owned(a) -> owned(b)
  }
  def min[S: Sc] = owned(ATen.min_2(value))

  /** Return a boolean tensor indicating elementwise min. */
  def min[S: Sc](other: STen) = owned(ATen.min_1(value, other.value))

  /** Reduces the given dimension with its max. */
  def min[S: Sc](dim: Int, keepDim: Boolean) = {
    val (a, b) = ATen.min_0(value, dim, keepDim)
    owned(a) -> owned(b)
  }

  /** Returns a long tensors with the argsort of the given dimension.
    *
    * Indexing the given dimension by the returned tensor would result in a sorted order.
    */
  def argsort[S: Sc](dim: Int, descending: Boolean) =
    owned(ATen.argsort(value, dim, descending))

  def cholesky[S: Sc](upper: Boolean) =
    owned(ATen.cholesky(value, upper))
  def choleskyInverse[S: Sc](upper: Boolean) =
    owned(ATen.cholesky_inverse(value, upper))
  def choleskyInverse[S: Sc](input2: STen, upper: Boolean) =
    owned(ATen.cholesky_solve(value, input2.value, upper))

  /** Return a boolean tensor indicating element-wise equality. Maps to Aten.equal */
  def equalDeep(input2: STen) =
    ATen.equal(value, input2.value)

  /** Return a boolean tensor indicating element-wise greater-than. */
  def gt[S: Sc](other: STen) =
    ATen.gt_1(value, other.value).owned

  /** Return a boolean tensor indicating element-wise greater-than. */
  def gt[S: Sc](other: Double) =
    ATen.gt_0(value, other).owned

  /** Return a boolean tensor indicating element-wise less-than. */
  def lt[S: Sc](other: STen) =
    ATen.lt_1(value, other.value).owned

  /** Return a boolean tensor indicating element-wise greater-than. */
  def lt[S: Sc](other: Double) =
    ATen.lt_0(value, other).owned

  /** Return a boolean tensor indicating element-wise greater-or-equal. */
  def ge[S: Sc](other: STen) =
    ATen.ge_1(value, other.value).owned

  /** Return a boolean tensor indicating element-wise greater-or-equal. */
  def ge[S: Sc](other: Double) =
    ATen.ge_0(value, other).owned

  /** Return a boolean tensor indicating element-wise less-or-equal. */
  def le[S: Sc](other: STen) =
    ATen.le_1(value, other.value).owned

  /** Return a boolean tensor indicating element-wise less-or-equal. */
  def le[S: Sc](other: Double) =
    ATen.le_0(value, other).owned

  /** Return a boolean tensor indicating element-wise not-equal. */
  def ne[S: Sc](other: STen) =
    ATen.ne_1(value, other.value).owned

  /** Return a boolean tensor indicating element-wise less-or-equal. */
  def ne[S: Sc](other: Double) =
    ATen.ne_0(value, other).owned

  /** Returns the negation. */
  def neg[S: Sc] =
    ATen.neg(value).owned

  /** Return a boolean tensor indicating element-wise is-nan. */
  def isnan[S: Sc] =
    ATen.isnan(value).owned

  /** Return a boolean tensor indicating element-wise is-finite. */
  def isfinite[S: Sc] =
    ATen.isfinite(value).owned
  def log10[S: Sc] =
    ATen.log10(value).owned
  def expm1[S: Sc] =
    ATen.expm1(value).owned

  /** Indexes with the given tensors along multiple dimensions. */
  def index[S: Sc](indices: STen*) =
    ATen.index(value, indices.map(_.value).toArray).owned
  def indexPut[S: Sc](indices: List[STen], values: STen, accumulate: Boolean) =
    ATen
      .index_put(value, indices.map(_.value).toArray, values.value, accumulate)
      .owned
  def indexCopy[S: Sc](dim: Int, index: STen, source: STen) =
    ATen
      .index_copy(value, dim, index.value, source.value)
      .owned
  def index_copy_(dim: Int, index: STen, source: STen): Unit =
    ATen
      ._index_copy_(value, dim, index.value, source.value)

  def matrixPower[S: Sc](n: Int) =
    ATen.matrix_power(value, n).owned
  def matrixRank[S: Sc](tol: Double, symmetric: Boolean) =
    ATen.matrix_rank(value, tol, symmetric).owned

  /** Returns a tensor with a subset of its elements.
    *
    * The returned tensor includes elements from `start` to `start+length` along the given dimension.
    *
    * No copy is made, storage is shared.
    */
  def narrow[S: Sc](dim: Int, start: Long, length: Long) =
    ATen.narrow_0(value, dim, start, length).owned
  def narrow[S: Sc](dim: Int, start: STen, length: Long) =
    ATen.narrow_1(value, dim, start.value, length).owned

  def oneHot[S: Sc](numClasses: Int) =
    ATen.one_hot(value, numClasses).owned

  def pinverse[S: Sc](rcond: Double) =
    ATen.pinverse(value, rcond).owned

  def repeatInterleave[S: Sc](repeats: STen, dim: Int) =
    ATen.repeat_interleave_1(value, repeats.value, dim).owned
  def repeatInterleave[S: Sc](repeats: Long, dim: Int) =
    ATen.repeat_interleave_2(value, repeats, dim).owned
  def repeatInterleave[S: Sc] =
    ATen.repeat_interleave_0(value).owned

  def repeat[S: Sc](dims: List[Long]) = value.repeat(dims.toArray).owned

  def sort[S: Sc](dim: Int, descending: Boolean) = {
    val (a, b) = ATen.sort(value, dim, descending)
    (owned(a), owned(b))
  }

  /** Removes dimensions of size=1 from the shape */
  def squeeze[S: Sc](dim: Int) =
    ATen.squeeze_1(value, dim).owned

  /** Removes dimensions of size=1 from the shape */
  def squeeze[S: Sc] =
    ATen.squeeze_0(value).owned

  /** Inserts a dimension of size=1 in the given position */
  def unsqueeze[S: Sc](dim: Int) =
    ATen.unsqueeze(value, dim).owned
  def varAndMean[S: Sc](unbiased: Boolean) = {
    val (a, b) = ATen.var_mean_0(value, unbiased)
    (a.owned, b.owned)
  }
  def varAndMean[S: Sc](dim: Seq[Int], unbiased: Boolean, keepDim: Boolean) = {
    val (a, b) =
      ATen.var_mean_1(value, dim.map(_.toLong).toArray, unbiased, keepDim)
    (a.owned, b.owned)
  }
  def stdAndMean[S: Sc](unbiased: Boolean) = {
    val (a, b) = ATen.std_mean_0(value, unbiased)
    (a.owned, b.owned)
  }
  def stdAndMean[S: Sc](dim: Seq[Int], unbiased: Boolean, keepDim: Boolean) = {
    val (a, b) =
      ATen.std_mean_1(value, dim.map(_.toLong).toArray, unbiased, keepDim)
    (a.owned, b.owned)
  }

  /** Returns the indices of non-zero values */
  def where[S: Sc] =
    ATen.where_1(value).toList.map(_.owned)

  def round[S: Sc] = ATen.round(value).owned

  def dropout_(p: Double, training: Boolean): Unit =
    ATen.dropout_(value, p, training)

  def frobeniusNorm[S: Sc] = ATen.frobenius_norm_0(value).owned

  def svd[S: Sc] = {
    val (ut, s, v) = ATen.svd(value, true, true)
    (ut.owned, s.owned, v.owned)
  }

  // todo:
  // ATen.eig
  // ATen.svd

}
