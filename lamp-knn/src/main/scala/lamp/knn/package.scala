package lamp

import org.saddle._
import org.saddle.macros.BinOps._
import lamp.autograd._
import aten.Tensor
import aten.ATen

package object knn {

  def squaredEuclideanDistance(t1: Tensor, t2: Tensor): Tensor = {
    withPool { implicit pool =>
      val v1 = t1.asVariable
      val v2 = t2.asVariable
      val outer = v1.mm(v2.t)
      val n1 = (v1 * v1).rowSum
      val n2 = (v2 * v2).rowSum
      (n1 + n2.t - outer * 2).keep.value
    }
  }

  def jaccardDistance(t1: Tensor, t2: Tensor): Tensor = {
    withPool { implicit pool =>
      val v1 = t1.asVariable
      val v2 = t2.asVariable
      val outer = v1.mm(v2.t)
      val n1 = v1.rowSum
      val n2 = v2.rowSum
      val denom = n1 + n2.t - outer
      val sim = outer / denom
      val one = const(1d, v1.options)(v1.pool)
      (one - sim).keep.value
    }
  }

  def knn(
      d: Tensor,
      query: Tensor,
      k: Int,
      distanceMatrix: (Tensor, Tensor) => Tensor
  ) = {
    val distance = distanceMatrix(query, d)
    val (topk1, topkindices) = ATen.topk(distance, k, 1, false, false)
    topk1.release
    distance.release
    topkindices
  }

  def knnMinibatched(
      d: Tensor,
      query: Tensor,
      k: Int,
      distanceMatrix: (Tensor, Tensor) => Tensor,
      minibatchSize: Int
  ) = {
    val rows = query.sizes.apply(0)
    val slices = ((0L until rows) grouped minibatchSize map { slice =>
      val first = slice.head
      val last = slice.last
      val querySlice = ATen.slice(query, 0, first, last + 1, 1)
      val indices = knn(d, querySlice, k, distanceMatrix)
      querySlice.release
      indices
    }).toArray
    val cat = ATen.cat(slices, 0)
    slices.foreach(_.release)
    cat
  }

  def regression(values: Vec[Double], indices: Mat[Int]): Vec[Double] =
    indices.rows.map(r => values.take(r.toArray).mean2).toVec

  def classification(
      values: Vec[Int],
      indices: Mat[Int],
      numClasses: Int
  ): Mat[Double] =
    Mat(indices.rows.map { r =>
      val selected = values.take(r.toArray)
      vec
        .range(0, numClasses)
        .map(c => selected.countif(_ == c).toDouble) / selected.length
    }: _*).T

  def knnSearch(
      features: Mat[Double],
      query: Mat[Double],
      k: Int,
      distance: (Tensor, Tensor) => Tensor,
      device: Device,
      precision: FloatingPointPrecision,
      minibatchSize: Int
  ): Mat[Int] = {
    val featuresOnDevice = TensorHelpers.fromMat(features, device, precision)
    val queryOnDevice = TensorHelpers.fromMat(query, device, precision)
    val indices = knnMinibatched(
      featuresOnDevice,
      queryOnDevice,
      k,
      distance,
      minibatchSize
    )
    val indicesJvm = TensorHelpers.toLongMat(indices).map(_.toInt)
    featuresOnDevice.release
    queryOnDevice.release
    indices.release
    indicesJvm
  }
  def knnClassification(
      features: Mat[Double],
      values: Vec[Int],
      query: Mat[Double],
      k: Int,
      distance: (Tensor, Tensor) => Tensor,
      device: Device,
      precision: FloatingPointPrecision,
      minibatchSize: Int
  ) = {
    val indices =
      knnSearch(features, query, k, distance, device, precision, minibatchSize)

    val numClasses = values.toArray.distinct.size
    classification(values, indices, numClasses)
  }
  def knnRegression(
      features: Mat[Double],
      values: Vec[Double],
      query: Mat[Double],
      k: Int,
      distance: (Tensor, Tensor) => Tensor = squaredEuclideanDistance _,
      device: Device = CPU,
      precision: FloatingPointPrecision = DoublePrecision,
      minibatchSize: Int = Int.MaxValue
  ) = {
    val indices =
      knnSearch(features, query, k, distance, device, precision, minibatchSize)

    regression(values, indices)
  }

}
