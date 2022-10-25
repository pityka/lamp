package lamp

import org.saddle._
import lamp.saddle._

package object knn {

  trait DistanceFunction {
    def apply(a: STen, b: STen)(implicit scope: Scope): STen
  }

  object SquaredEuclideanDistance extends DistanceFunction {
    def apply(a: STen, b: STen)(implicit scope: Scope): STen =
      squaredEuclideanDistance(a, b)
  }
  object JaccardDistance extends DistanceFunction {
    def apply(a: STen, b: STen)(implicit scope: Scope): STen =
      jaccardDistance(a, b)
  }

  def squaredEuclideanDistance(v1: STen, v2: STen)(implicit
      scope: Scope
  ): STen = {
    Scope { implicit scope =>
      val outer = v1.mm(v2.t)
      val n1 = (v1 * v1).rowSum
      val n2 = (v2 * v2).rowSum
      (n1 + n2.t - outer * 2).max(STen.zeros(List(1),v1.options))
    }
  }

  def jaccardDistance(v1: STen, v2: STen)(implicit
      scope: Scope
  ): STen = {
    Scope { implicit scope =>
      val outer = v1.mm(v2.t)
      val n1 = v1.rowSum
      val n2 = v2.rowSum
      val denom = n1 + n2.t - outer
      val sim = outer / denom
      val one = STen.ones(List(1), sim.options)
      (one - sim)
    }
  }

  def knn(
      d: STen,
      query: STen,
      k: Int,
      distanceMatrix: DistanceFunction
  )(implicit scope: Scope) = {
    Scope { implicit scope =>
      val distance =
        distanceMatrix(query, d)
      val (_, topkindices) = distance.topk(k, 1, false, false)
      topkindices
    }
  }

  def knnMinibatched(
      d: STen,
      query: STen,
      k: Int,
      distanceMatrix: DistanceFunction,
      minibatchSize: Int
  )(implicit scope: Scope) = {
    Scope { implicit scope =>
      val rows = query.shape.apply(0)
      val slices = ((0L until rows) grouped minibatchSize map { slice =>
        Scope { implicit scope =>
          val first = slice.head
          val last = slice.last
          val querySlice = query.slice(0, first, last + 1, 1)
          val indices = knn(d, querySlice, k, distanceMatrix)
          indices
        }
      }).toList
      STen.cat(slices, 0)
    }
  }

  def regression(values: Vec[Double], indices: Mat[Int]): Vec[Double] =
    indices.rows.map(r => values.take(r.toArray).mean2).toVec

  def classification(
      values: Vec[Int],
      indices: Mat[Int],
      numClasses: Int,
      log: Boolean
  ): Mat[Double] =
    Mat(indices.rows.map { r =>
      val selected = values.take(r.toArray)
      vec
        .range(0, numClasses)
        .map(c => selected.countif(_ == c).toDouble).map(_ / selected.length)
    }: _*).T.map(v => if (log) math.log(v + 1e-6) else v)

  def knnSearch(
      features: Mat[Double],
      query: Mat[Double],
      k: Int,
      distance: DistanceFunction,
      device: Device,
      precision: FloatingPointPrecision,
      minibatchSize: Int
  ): Mat[Int] = {
    Scope.root { implicit scope =>
      val featuresOnDevice = lamp.saddle.fromMat(features, device, precision)
      val queryOnDevice = lamp.saddle.fromMat(query, device, precision)
      val indices = knnMinibatched(
        featuresOnDevice,
        queryOnDevice,
        k,
        distance,
        minibatchSize
      )

      val indicesJvm = indices.toLongMat.map(_.toInt)
      indicesJvm
    }
  }
  def knnClassification(
      features: Mat[Double],
      values: Vec[Int],
      query: Mat[Double],
      k: Int,
      distance: DistanceFunction,
      device: Device,
      precision: FloatingPointPrecision,
      minibatchSize: Int,
      log: Boolean
  ) = {
    val indices =
      knnSearch(features, query, k, distance, device, precision, minibatchSize)

    val numClasses = values.toArray.distinct.size
    classification(values, indices, numClasses, log)
  }
  def knnRegression(
      features: Mat[Double],
      values: Vec[Double],
      query: Mat[Double],
      k: Int,
      distance: DistanceFunction,
      device: Device = CPU,
      precision: FloatingPointPrecision = DoublePrecision,
      minibatchSize: Int = Int.MaxValue
  ) = {
    val indices =
      knnSearch(features, query, k, distance, device, precision, minibatchSize)

    regression(values, indices)
  }

}
