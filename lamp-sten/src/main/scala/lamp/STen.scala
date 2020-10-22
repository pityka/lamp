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
      cuda: Boolean
  ) = owned(TensorHelpers.fromMat(m, cuda))
  def fromMat[S: Sc](
      m: Mat[Double],
      device: Device,
      precision: FloatingPointPrecision
  ) = owned(TensorHelpers.fromMat(m, device, precision))

  def fromLongMat[S: Sc](
      m: Mat[Long],
      device: Device
  ) = owned(TensorHelpers.fromLongMat(m, device))
  def fromLongMat[S: Sc](
      m: Mat[Long],
      cuda: Boolean
  ) = owned(TensorHelpers.fromLongMat(m, cuda))

  def free(value: Tensor) = STen(value, Scope.free)

  def owned(
      value: Tensor
  )(implicit scope: Scope): STen =
    STen(value, scope)

  def addOut(
      out: STen,
      self: STen,
      other: STen,
      alpha: Double
  ): Unit =
    ATen.add_out(out.value, self.value, other.value, alpha)
  def zeroInplace(
      self: STen
  ): Unit =
    ATen.zero_(self.value)

  def cat[S: Sc](tensors: Seq[STen], dim: Long) =
    owned(ATen.cat(tensors.map(_.value).toArray, dim))

  def ones[S: Sc](
      size: List[Int],
      tensorOptions: TensorOptions
  ) =
    owned(ATen.ones(size.toArray.map(_.toLong), tensorOptions))
  def zeros[S: Sc](size: List[Int], tensorOptions: TensorOptions) =
    owned(ATen.zeros(size.toArray.map(_.toLong), tensorOptions))
  def rand[S: Sc](size: List[Int], tensorOptions: TensorOptions) =
    owned(ATen.rand(size.toArray.map(_.toLong), tensorOptions))
  def randn[S: Sc](size: List[Int], tensorOptions: TensorOptions) =
    owned(ATen.randn(size.toArray.map(_.toLong), tensorOptions))
  def randint[S: Sc](
      high: Long,
      size: List[Int],
      tensorOptions: TensorOptions
  ) =
    owned(
      ATen.randint_0(high, size.toArray.map(_.toLong), tensorOptions)
    )
  def eye[S: Sc](n: Int, tensorOptions: TensorOptions) =
    owned(ATen.eye_0(n, tensorOptions))
  def eye[S: Sc](n: Int, m: Int, tensorOptions: TensorOptions) =
    owned(ATen.eye_1(n, m, tensorOptions))

  def arange[S: Sc](
      start: Double,
      end: Double,
      step: Double,
      tensorOptions: TensorOptions
  ) =
    owned(ATen.arange(start, end, step, tensorOptions))
  def linspace[S: Sc](
      start: Double,
      end: Double,
      steps: Long,
      tensorOptions: TensorOptions
  ) =
    owned(ATen.linspace(start, end, steps, tensorOptions))
}

case class STen(
    value: Tensor,
    scope: Scope
) {
  import STen._
  scope(value)

  def toMat = TensorHelpers.toMat(value)
  def toLongMat = TensorHelpers.toLongMat(value)
  def toVec = toMat.toVec
  val shape = value.sizes.toList
  def sizes = shape
  val options = value.options()

  def copyTo(implicit scope: Scope, device: Device) = {
    STen(device.to(value), scope)
  }

  override def toString =
    s"STen(shape=$shape,value=$value,scope=$scope)"

  def t[S: Sc] = owned(ATen.t(value))
  def transpose[S: Sc](dim1: Int, dim2: Int) =
    owned(ATen.transpose(value, dim1, dim2))

  def select[S: Sc](dim: Long, index: Long) =
    owned(ATen.select(value, dim, index))
  def indexSelect[S: Sc](dim: Long, index: STen) =
    owned(ATen.index_select(value, dim, index.value))
  def argmax[S: Sc](dim: Long, keepDim: Boolean) =
    owned(ATen.argmax(value, dim, keepDim))
  def argmin[S: Sc](dim: Long, keepDim: Boolean) =
    owned(ATen.argmin(value, dim, keepDim))
  def maskFill[S: Sc](mask: STen, fill: Double) =
    owned(ATen.masked_fill_0(value, mask.value, fill))
  def eq[S: Sc](other: STen) =
    owned(ATen.eq_1(value, other.value))
  def cat[S: Sc](other: STen, dim: Long) =
    owned(ATen.cat(Array(value, other.value), dim))

  def castToFloat[S: Sc] = owned(ATen._cast_Float(value, false))
  def castToDouble[S: Sc] = owned(ATen._cast_Double(value, false))
  def castToLong[S: Sc] = owned(ATen._cast_Long(value, false))

  def +[S: Sc](other: STen) =
    owned(ATen.add_0(value, other.value, 1d))
  def add[S: Sc](other: STen, alpha: Double) =
    owned(ATen.add_0(value, other.value, alpha))
  def add[S: Sc](other: Double, alpha: Double) =
    owned(ATen.add_1(value, other, alpha))

  def -[S: Sc](other: STen) =
    owned(ATen.sub_0(value, other.value, 1d))
  def sub[S: Sc](other: STen, alpha: Double) =
    owned(ATen.sub_0(value, other.value, alpha))
  def sub[S: Sc](other: Double, alpha: Double) =
    owned(ATen.sub_1(value, other, alpha))

  def *[S: Sc](other: STen) =
    owned(ATen.mul_0(value, other.value))
  def *[S: Sc](other: Double) =
    owned(ATen.mul_1(value, other))

  def /[S: Sc](other: STen) =
    owned(ATen.div_0(value, other.value))
  def /[S: Sc](other: Double) =
    owned(ATen.div_1(value, other))

  def mm[S: Sc](other: STen) =
    owned(ATen.mm(value, other.value))
  def bmm[S: Sc](other: STen) =
    owned(ATen.bmm(value, other.value))

  def relu[S: Sc] = owned(ATen.relu(value))
  def gelu[S: Sc] = owned(ATen.gelu(value))
  def sigmoid[S: Sc] = owned(ATen.sigmoid(value))
  def exp[S: Sc] = owned(ATen.exp(value))
  def log[S: Sc] = owned(ATen.log(value))
  def log1p[S: Sc] = owned(ATen.log1p(value))
  def sin[S: Sc] = owned(ATen.sin(value))
  def cos[S: Sc] = owned(ATen.cos(value))
  def tan[S: Sc] = owned(ATen.tan(value))
  def tanh[S: Sc] = owned(ATen.tanh(value))
  def atanh[S: Sc] = owned(ATen.atan(value))
  def acos[S: Sc] = owned(ATen.acos(value))
  def asin[S: Sc] = owned(ATen.asin(value))
  def sqrt[S: Sc] = owned(ATen.sqrt(value))
  def square[S: Sc] = owned(ATen.square(value))
  def abs[S: Sc] = owned(ATen.abs(value))
  def ceil[S: Sc] = owned(ATen.ceil(value))
  def floor[S: Sc] = owned(ATen.floor(value))
  def reciprocal[S: Sc] = owned(ATen.reciprocal(value))
  def det[S: Sc] = owned(ATen.det(value))
  def trace[S: Sc] = owned(ATen.trace(value))

  def remainder[S: Sc](other: STen) =
    ATen.remainder_1(value, other.value).owned
  def remainder[S: Sc](other: Double) =
    ATen.remainder_0(value, other).owned

  def pow[S: Sc](exponent: Double) = owned(ATen.pow_0(value, exponent))
  def pow[S: Sc](exponent: STen) =
    owned(ATen.pow_1(value, exponent.value))

  def sum[S: Sc] = owned(ATen.sum_0(value))
  def sum[S: Sc](dim: List[Int], keepDim: Boolean) =
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

  def expandAs[S: Sc](other: STen) =
    owned(value.expand_as(other.value))
  def view[S: Sc](other: List[Int]) =
    owned(ATen._unsafe_view(value, other.map(_.toLong).toArray))

  def norm2[S: Sc](dim: List[Int], keepDim: Boolean) =
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
  def mean[S: Sc](dim: List[Int], keepDim: Boolean) =
    owned(ATen.mean_1(value, dim.toArray.map(_.toLong), keepDim))
  def mean[S: Sc](dim: Int, keepDim: Boolean) =
    owned(ATen.mean_1(value, Array(dim), keepDim))

  def variance[S: Sc](unbiased: Boolean) =
    owned(ATen.var_0(value, unbiased))
  def variance[S: Sc](dim: List[Int], unbiased: Boolean, keepDim: Boolean) =
    owned(ATen.var_1(value, dim.toArray.map(_.toLong), unbiased, keepDim))
  def variance[S: Sc](dim: Int, unbiased: Boolean, keepDim: Boolean) =
    owned(ATen.var_1(value, Array(dim), unbiased, keepDim))

  def std[S: Sc](unbiased: Boolean) =
    owned(ATen.std_0(value, unbiased))
  def std[S: Sc](dim: List[Int], unbiased: Boolean, keepDim: Boolean) =
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
  def index[S: Sc](indices: List[STen]) =
    ATen.index(value, indices.map(_.value).toArray).owned
  def indexPut[S: Sc](indices: List[STen], values: STen, accumulate: Boolean) =
    ATen
      .index_put(value, indices.map(_.value).toArray, values.value, accumulate)
      .owned
  def indexCopy[S: Sc](dim: Int, index: STen, source: STen) =
    ATen
      .index_copy(value, dim, index.value, source.value)
      .owned

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
  def varAndMean[S: Sc](dim: List[Int], unbiased: Boolean, keepDim: Boolean) = {
    val (a, b) =
      ATen.var_mean_1(value, dim.map(_.toLong).toArray, unbiased, keepDim)
    (a.owned, b.owned)
  }
  def stdAndMean[S: Sc](unbiased: Boolean) = {
    val (a, b) = ATen.std_mean_0(value, unbiased)
    (a.owned, b.owned)
  }
  def stdAndMean[S: Sc](dim: List[Int], unbiased: Boolean, keepDim: Boolean) = {
    val (a, b) =
      ATen.std_mean_1(value, dim.map(_.toLong).toArray, unbiased, keepDim)
    (a.owned, b.owned)
  }

  def where[S: Sc](condition: STen, other: STen) =
    ATen.where_0(condition.value, value, other.value).owned
  def where[S: Sc] =
    ATen.where_1(value).toList.map(_.owned)

  // todo:
  // ATen.eig
  // ATen.svd

}
