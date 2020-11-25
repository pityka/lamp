package lamp

import aten.Tensor
import aten.ATen
import aten.TensorOptions
import org.saddle._

object STen {

  implicit class OwnedSyntax(t: Tensor) {
    def owned[S: Sc] = STen.owned(t)
  }

  def fromMat[S: Sc](
      m: Mat[Double],
      cuda: Boolean = false
  ) = owned(TensorHelpers.fromMat(m, cuda))
  def fromMat[S: Sc](
      m: Mat[Double],
      device: Device,
      precision: FloatingPointPrecision
  ) = owned(TensorHelpers.fromMat(m, device, precision))
  def fromFloatMat[S: Sc](
      m: Mat[Float],
      device: Device
  ) = owned(TensorHelpers.fromFloatMat(m, device))
  def fromVec[S: Sc](
      m: Vec[Double],
      cuda: Boolean = false
  ) = owned(TensorHelpers.fromVec(m, cuda))
  def fromVec[S: Sc](
      m: Vec[Double],
      device: Device,
      precision: FloatingPointPrecision
  ) = owned(TensorHelpers.fromVec(m, device, precision))

  def fromLongMat[S: Sc](
      m: Mat[Long],
      device: Device
  ) = owned(TensorHelpers.fromLongMat(m, device))
  def fromLongMat[S: Sc](
      m: Mat[Long],
      cuda: Boolean = false
  ) = owned(TensorHelpers.fromLongMat(m, cuda))

  def fromLongVec[S: Sc](
      m: Vec[Long],
      device: Device
  ) = owned(TensorHelpers.fromLongVec(m, device))
  def fromLongVec[S: Sc](
      m: Vec[Long],
      cuda: Boolean = false
  ) = owned(TensorHelpers.fromLongVec(m, cuda))

  def fromLongArray[S: Sc](ar: Array[Long], dim: Seq[Long], device: Device) =
    TensorHelpers.fromLongArray(ar, dim, device).owned
  def fromDoubleArray[S: Sc](
      ar: Array[Double],
      dim: Seq[Long],
      device: Device,
      precision: FloatingPointPrecision
  ) =
    TensorHelpers.fromDoubleArray(ar, dim, device, precision).owned
  def fromFloatArray[S: Sc](ar: Array[Float], dim: Seq[Long], device: Device) =
    TensorHelpers.fromFloatArray(ar, dim, device).owned

  def free(value: Tensor) = STen(value)

  def apply[S: Sc](vs: Double*) = fromVec(Vec(vs: _*))

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

  def scalarLong(value: Long, options: TensorOptions)(implicit scope: Scope) =
    Tensor.scalarLong(value, options.toLong()).owned
  def scalarDouble[S: Sc](value: Double, options: TensorOptions) =
    Tensor.scalarDouble(value, options.toDouble()).owned

  def ones[S: Sc](
      size: Seq[Long],
      tensorOptions: TensorOptions = TensorOptions.d
  ) =
    owned(ATen.ones(size.toArray.map(_.toLong), tensorOptions))
  def zeros[S: Sc](
      size: Seq[Long],
      tensorOptions: TensorOptions = TensorOptions.d
  ) =
    owned(ATen.zeros(size.toArray.map(_.toLong), tensorOptions))
  def rand[S: Sc](
      size: Seq[Long],
      tensorOptions: TensorOptions = TensorOptions.d
  ) =
    owned(ATen.rand(size.toArray.map(_.toLong), tensorOptions))
  def randn[S: Sc](
      size: Seq[Long],
      tensorOptions: TensorOptions = TensorOptions.d
  ) =
    owned(ATen.randn(size.toArray.map(_.toLong), tensorOptions))

  def normal[S: Sc](
      mean: Double,
      std: Double,
      size: Seq[Long],
      options: TensorOptions
  ) =
    ATen.normal_3(mean, std, size.toArray, options).owned

  def randint[S: Sc](
      high: Long,
      size: Seq[Long],
      tensorOptions: TensorOptions = TensorOptions.d
  ) =
    owned(
      ATen.randint_0(high, size.toArray, tensorOptions)
    )
  def randint[S: Sc](
      low: Long,
      high: Long,
      size: Seq[Long],
      tensorOptions: TensorOptions
  ) =
    owned(
      ATen.randint_2(low, high, size.toArray, tensorOptions)
    )
  def eye[S: Sc](n: Int, tensorOptions: TensorOptions = TensorOptions.d) =
    owned(ATen.eye_0(n, tensorOptions))
  def eye[S: Sc](
      n: Int,
      m: Int,
      tensorOptions: TensorOptions
  ) =
    owned(ATen.eye_1(n, m, tensorOptions))

  def arange[S: Sc](
      start: Double,
      end: Double,
      step: Double,
      tensorOptions: TensorOptions = TensorOptions.d
  ) =
    owned(ATen.arange(start, end, step, tensorOptions))
  def linspace[S: Sc](
      start: Double,
      end: Double,
      steps: Long,
      tensorOptions: TensorOptions = TensorOptions.d
  ) =
    owned(ATen.linspace(start, end, steps, tensorOptions))
  def sparse_coo[S: Sc](
      indices: STen,
      values: STen,
      dim: Seq[Long],
      tensorOptions: TensorOptions = TensorOptions.d
  ) =
    owned(
      ATen.sparse_coo_tensor(
        indices.value,
        values.value,
        dim.toArray,
        tensorOptions
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

case class STen private (
    value: Tensor
) {
  import STen._

  def numel = value.numel

  def toMat = TensorHelpers.toMat(value)
  def toFloatMat = TensorHelpers.toFloatMat(value)
  def toLongMat = TensorHelpers.toLongMat(value)
  def toVec = TensorHelpers.toVec(value)
  def toFloatVec = TensorHelpers.toFloatVec(value)
  def toLongVec = TensorHelpers.toLongVec(value)
  val shape = value.sizes.toList
  def sizes = shape
  val options = value.options()
  def coalesce[S: Sc] = value.coalesce.owned
  def indices[S: Sc] = value.indices.owned
  def values[S: Sc] = value.indices.owned

  def copyToDevice(device: Device)(implicit scope: Scope) = {
    STen.owned(device.to(value))
  }

  def cloneTensor[S: Sc] = ATen.clone(value).owned
  def copyTo[S: Sc](options: TensorOptions) = value.to(options, true).owned
  def copyFrom(source: Tensor) =
    value.copyFrom(source)
  def copyFrom(source: STen) =
    value.copyFrom(source.value)

  override def toString =
    s"STen(shape=$shape,value=$value)"

  def unbroadcast[S: Sc](sizes: Seq[Long]) =
    TensorHelpers.unbroadcast(value, sizes.toList) match {
      case None    => this
      case Some(t) => t.owned
    }

  def t[S: Sc] = owned(ATen.t(value))
  def transpose[S: Sc](dim1: Int, dim2: Int) =
    owned(ATen.transpose(value, dim1, dim2))

  def select[S: Sc](dim: Long, index: Long) =
    owned(ATen.select(value, dim, index))
  def flatten[S: Sc](startDim: Long, endDim: Long) =
    owned(ATen.flatten(value, startDim, endDim))

  def indexSelect[S: Sc](dim: Long, index: STen) =
    owned(ATen.index_select(value, dim, index.value))
  def indexSelect[S: Sc](dim: Long, index: Tensor) =
    owned(ATen.index_select(value, dim, index))
  def argmax[S: Sc](dim: Long, keepDim: Boolean) =
    owned(ATen.argmax(value, dim, keepDim))
  def argmin[S: Sc](dim: Long, keepDim: Boolean) =
    owned(ATen.argmin(value, dim, keepDim))
  def maskFill[S: Sc](mask: STen, fill: Double) =
    owned(ATen.masked_fill_0(value, mask.value, fill))
  def equ[S: Sc](other: STen) =
    owned(ATen.eq_1(value, other.value))
  def equ[S: Sc](other: Double) =
    owned(ATen.eq_0(value, other))
  def cat[S: Sc](other: STen, dim: Long) =
    owned(ATen.cat(Array(value, other.value), dim))

  def castToFloat[S: Sc] = owned(ATen._cast_Float(value, false))
  def castToDouble[S: Sc] = owned(ATen._cast_Double(value, false))
  def castToLong[S: Sc] = owned(ATen._cast_Long(value, false))

  def +[S: Sc](other: STen) =
    owned(ATen.add_0(value, other.value, 1d))
  def +=(other: STen): Unit =
    ATen.add_out(value, value, other.value, 1d)
  def +[S: Sc](other: Double) =
    owned(ATen.add_1(value, other, 1d))
  def add[S: Sc](other: STen, alpha: Double) =
    owned(ATen.add_0(value, other.value, alpha))
  def add[S: Sc](other: Double, alpha: Double) =
    owned(ATen.add_1(value, other, alpha))

  def addmm[S: Sc](mat1: STen, mat2: STen, beta: Double, alpha: Double) =
    ATen.addmm(value, mat1.value, mat2.value, beta, alpha).owned

  def -[S: Sc](other: STen) =
    owned(ATen.sub_0(value, other.value, 1d))
  def -=[S: Sc](other: STen): Unit =
    ATen.sub_out(value, value, other.value, 1d)
  def sub[S: Sc](other: STen, alpha: Double) =
    owned(ATen.sub_0(value, other.value, alpha))
  def sub[S: Sc](other: Double, alpha: Double) =
    owned(ATen.sub_1(value, other, alpha))

  def *[S: Sc](other: STen) =
    owned(ATen.mul_0(value, other.value))
  def *[S: Sc](other: Tensor) =
    owned(ATen.mul_0(value, other))

  def *[S: Sc](other: Double) =
    owned(ATen.mul_1(value, other))

  def *=[S: Sc](other: STen): Unit =
    ATen.mul_out(value, value, other.value)

  def /[S: Sc](other: STen) =
    owned(ATen.div_0(value, other.value))
  def /[S: Sc](other: Tensor) =
    owned(ATen.div_0(value, other))
  def /[S: Sc](other: Double) =
    owned(ATen.div_1(value, other))
  def /=[S: Sc](other: STen): Unit =
    ATen.div_out(value, value, other.value)

  def mm[S: Sc](other: STen) =
    owned(ATen.mm(value, other.value))

  def bmm[S: Sc](other: STen) =
    owned(ATen.bmm(value, other.value))

  def baddbmm[S: Sc](batch1: STen, batch2: STen, beta: Double, alpha: Double) =
    ATen.baddbmm(value, batch1.value, batch2.value, beta, alpha).owned

  def addcmul[S: Sc](
      tensor1: STen,
      tensor2: STen,
      alpha: Double
  ) =
    ATen.addcmul(value, tensor1.value, tensor2.value, alpha).owned
  def addcmulSelf(
      tensor1: STen,
      tensor2: STen,
      alpha: Double
  ): Unit =
    ATen.addcmul_out(value, value, tensor1.value, tensor2.value, alpha)
  def addcmulSelf(
      tensor1: STen,
      tensor2: Tensor,
      alpha: Double
  ): Unit =
    ATen.addcmul_out(value, value, tensor1.value, tensor2, alpha)

  def relu[S: Sc] = owned(ATen.relu(value))
  def relu_() = ATen.relu_(value)
  def gelu[S: Sc] = owned(ATen.gelu(value))
  def sigmoid[S: Sc] = owned(ATen.sigmoid(value))
  def sigmoid_() = ATen.sigmoid_(value)
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
  def sum[S: Sc](dim: Seq[Int], keepDim: Boolean) =
    owned(ATen.sum_1(value, dim.toArray.map(_.toLong), keepDim))
  def sum[S: Sc](dim: Int, keepDim: Boolean) =
    owned(ATen.sum_1(value, Array(dim.toLong), keepDim))
  def rowSum[S: Sc] = sum(1, true)
  def colSum[S: Sc] = sum(0, true)

  def topk[S: Sc](k: Int, dim: Int, largest: Boolean, sorted: Boolean) = {
    val (a, b) = ATen.topk(value, k, dim, largest, sorted)
    (owned(a), owned(b))
  }

  def slice[S: Sc](dim: Int, start: Long, end: Long, step: Long) =
    owned(ATen.slice(value, dim, start, end, step))
  def slice[S: Sc](dim: Long, start: Long, end: Long, step: Long) =
    owned(ATen.slice(value, dim, start, end, step))

  def fill_(v: Double) = ATen.fill__0(value, v)

  def maskedSelect[S: Sc](mask: STen) =
    ATen.masked_select(value, mask.value).owned
  def maskedSelect[S: Sc](mask: Tensor) =
    ATen.masked_select(value, mask).owned

  def maskedFill[S: Sc](mask: STen, fill: Double) =
    ATen.masked_fill_0(value, mask.value, fill).owned
  def maskedFill[S: Sc](mask: Tensor, fill: Double) =
    ATen.masked_fill_0(value, mask, fill).owned

  def zero_(): Unit =
    ATen.zero_(value)

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
  def view[S: Sc](dims: Long*) =
    owned(ATen._unsafe_view(value, dims.toArray))

  def norm2[S: Sc](dim: Seq[Int], keepDim: Boolean) =
    owned(
      ATen.norm_2(
        value,
        dim.toArray.map(_.toLong),
        keepDim,
        TensorOptions.d.scalarTypeByte()
      )
    )

  def logSoftMax[S: Sc](dim: Int) =
    owned(
      ATen.log_softmax(
        value,
        dim
      )
    )

  def mean[S: Sc] =
    owned(ATen.mean_0(value))
  def mean[S: Sc](dim: Seq[Int], keepDim: Boolean) =
    owned(ATen.mean_1(value, dim.toArray.map(_.toLong), keepDim))
  def mean[S: Sc](dim: Int, keepDim: Boolean) =
    owned(ATen.mean_1(value, Array(dim), keepDim))

  def variance[S: Sc](unbiased: Boolean) =
    owned(ATen.var_0(value, unbiased))
  def variance[S: Sc](dim: Seq[Int], unbiased: Boolean, keepDim: Boolean) =
    owned(ATen.var_1(value, dim.toArray.map(_.toLong), unbiased, keepDim))
  def variance[S: Sc](dim: Int, unbiased: Boolean, keepDim: Boolean) =
    owned(ATen.var_1(value, Array(dim), unbiased, keepDim))

  def std[S: Sc](unbiased: Boolean) =
    owned(ATen.std_0(value, unbiased))
  def std[S: Sc](dim: Seq[Int], unbiased: Boolean, keepDim: Boolean) =
    owned(ATen.std_1(value, dim.toArray.map(_.toLong), unbiased, keepDim))
  def std[S: Sc](dim: Int, unbiased: Boolean, keepDim: Boolean) =
    owned(ATen.std_1(value, Array(dim), unbiased, keepDim))

  def median[S: Sc] = owned(ATen.median_1(value))
  def median[S: Sc](dim: Int, keepDim: Boolean) = {
    val (a, b) = ATen.median_0(value, dim, keepDim)
    owned(a) -> owned(b)
  }
  def mode[S: Sc](dim: Int, keepDim: Boolean) = {
    val (a, b) = ATen.mode(value, dim, keepDim)
    owned(a) -> owned(b)
  }

  def max[S: Sc] = owned(ATen.max_2(value))
  def max[S: Sc](other: STen) = owned(ATen.max_1(value, other.value))
  def max[S: Sc](dim: Int, keepDim: Boolean) = {
    val (a, b) = ATen.max_0(value, dim, keepDim)
    owned(a) -> owned(b)
  }
  def min[S: Sc] = owned(ATen.min_2(value))
  def min[S: Sc](other: STen) = owned(ATen.min_1(value, other.value))
  def min[S: Sc](dim: Int, keepDim: Boolean) = {
    val (a, b) = ATen.min_0(value, dim, keepDim)
    owned(a) -> owned(b)
  }

  def argsort[S: Sc](dim: Int, descending: Boolean) =
    owned(ATen.argsort(value, dim, descending))

  def cholesky[S: Sc](upper: Boolean) =
    owned(ATen.cholesky(value, upper))
  def choleskyInverse[S: Sc](upper: Boolean) =
    owned(ATen.cholesky_inverse(value, upper))
  def choleskyInverse[S: Sc](input2: STen, upper: Boolean) =
    owned(ATen.cholesky_solve(value, input2.value, upper))
  def equalDeep(input2: STen) =
    ATen.equal(value, input2.value)

  def gt[S: Sc](other: STen) =
    ATen.gt_1(value, other.value).owned
  def gt[S: Sc](other: Double) =
    ATen.gt_0(value, other).owned
  def lt[S: Sc](other: STen) =
    ATen.lt_1(value, other.value).owned
  def lt[S: Sc](other: Double) =
    ATen.lt_0(value, other).owned

  def ge[S: Sc](other: STen) =
    ATen.ge_1(value, other.value).owned
  def ge[S: Sc](other: Double) =
    ATen.ge_0(value, other).owned
  def le[S: Sc](other: STen) =
    ATen.le_1(value, other.value).owned
  def le[S: Sc](other: Double) =
    ATen.le_0(value, other).owned
  def ne[S: Sc](other: STen) =
    ATen.ne_1(value, other.value).owned
  def ne[S: Sc](other: Double) =
    ATen.ne_0(value, other).owned
  def neg[S: Sc] =
    ATen.neg(value).owned
  def isnan[S: Sc] =
    ATen.isnan(value).owned
  def isfinite[S: Sc] =
    ATen.isfinite(value).owned
  def log10[S: Sc] =
    ATen.log10(value).owned
  def expm1[S: Sc] =
    ATen.expm1(value).owned
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

  def sort[S: Sc](dim: Int, descending: Boolean) = {
    val (a, b) = ATen.sort(value, dim, descending)
    (owned(a), owned(b))
  }
  def squeeze[S: Sc](dim: Int) =
    ATen.squeeze_1(value, dim).owned
  def squeeze[S: Sc] =
    ATen.squeeze_0(value).owned
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

  def where[S: Sc] =
    ATen.where_1(value).toList.map(_.owned)

  def round[S: Sc] = ATen.round(value).owned

  def dropout_(p: Double, training: Boolean): Unit =
    ATen.dropout_(value, p, training)

  def frobeniusNorm[S: Sc] = ATen.frobenius_norm_0(value).owned

  // todo:
  // ATen.eig
  // ATen.svd

}
