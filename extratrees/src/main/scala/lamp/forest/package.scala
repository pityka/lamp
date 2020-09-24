package lamp

import org.saddle._
import org.saddle.macros.BinOps._
import java.util.concurrent.ForkJoinPool
import scala.concurrent.ExecutionContext
import cats.effect.IO
import cats.syntax.parallel._
import cats.data.NonEmptyList
import scala.reflect.ClassTag

sealed trait ClassificationTree
case class ClassificationLeaf(targetDistribution: Seq[Double])
    extends ClassificationTree
object ClassificationLeaf {
  import upickle.default.{ReadWriter => RW, macroRW}
  implicit val rw: RW[ClassificationLeaf] = macroRW
}
case class ClassificationNonLeaf(
    left: ClassificationTree,
    right: ClassificationTree,
    splitFeature: Int,
    cutpoint: Double
) extends ClassificationTree
object ClassificationNonLeaf {
  import upickle.default.{ReadWriter => RW, macroRW}
  implicit val rw: RW[ClassificationNonLeaf] = macroRW
}

object ClassificationTree {
  import upickle.default.{ReadWriter => RW, macroRW}
  implicit val rw: RW[ClassificationTree] = macroRW
}

sealed trait RegressionTree
case class RegressionLeaf(targetMean: Double) extends RegressionTree
case class RegressionNonLeaf(
    left: RegressionTree,
    right: RegressionTree,
    splitFeature: Int,
    cutpoint: Double
) extends RegressionTree
object RegressionLeaf {
  import upickle.default.{ReadWriter => RW, macroRW}
  implicit val rw: RW[RegressionLeaf] = macroRW
}

object RegressionNonLeaf {
  import upickle.default.{ReadWriter => RW, macroRW}
  implicit val rw: RW[RegressionNonLeaf] = macroRW
}

object RegressionTree {
  import upickle.default.{ReadWriter => RW, macroRW}
  implicit val rw: RW[RegressionTree] = macroRW
}

package object extratrees {

  def minmax(self: Vec[Double]) = {
    var sMin = Double.MaxValue
    var sMax = Double.MinValue
    var i = 0
    val n = self.length
    while (i < n) {
      val v = self.raw(i)
      if (v < sMin) {
        sMin = v
      }
      if (v > sMax) {
        sMax = v
      }
      i += 1
    }
    (sMin, sMax)
  }

  def splitClassification(
      data: Mat[Double],
      subset: Vec[Int],
      attributes: Array[Int],
      numConstant: Int,
      k: Int,
      targetAtSubset: Vec[Int],
      weightsAtSubset: Option[Vec[Double]],
      rng: org.saddle.spire.random.Generator,
      numClasses: Int
  ) = {
    val giniTotal = giniImpurity(targetAtSubset, weightsAtSubset, numClasses)
    val buf1 = Array.ofDim[Double](numClasses)
    val buf2 = Array.ofDim[Double](numClasses)

    var low = numConstant
    var high = attributes.length
    val N = attributes.length
    def swap(i: Int, j: Int) = {
      val t = attributes(i)
      attributes(i) = attributes(j)
      attributes(j) = t
    }

    var bestScore = Double.NegativeInfinity
    var bestFeature = -1
    var bestCutpoint = Double.NaN
    var visited = 0
    while (N - high < k && high - low > 0) {
      val r = rng.nextInt(low, high - 1)
      val attr = attributes(r)
      val (min, max) = minmax(takeCol(data, subset, attr))
      if (max <= min) {
        swap(r, low)
        low += 1
      } else {
        val cutpoint = rng.nextDouble(min, max)

        val take = takeCol(data, subset, attr) < cutpoint

        val score = giniScore(
          targetAtSubset,
          weightsAtSubset,
          take,
          giniTotal,
          numClasses,
          buf1,
          buf2
        )
        if (score > bestScore) {
          bestScore = score
          bestFeature = attr
          bestCutpoint = cutpoint
        }
        if (score.isNaN) {
          swap(r, low)
          low += 1
        } else {
          visited += 1
          swap(r, high - 1)
          high -= 1
        }
      }
    }
    if (visited == 0) (-1, bestCutpoint, low)
    else
      (bestFeature, bestCutpoint, low)
  }
  def splitRegression(
      data: Mat[Double],
      subset: Vec[Int],
      attributes: Array[Int],
      numConstant: Int,
      k: Int,
      targetAtSubset: Vec[Double],
      rng: org.saddle.spire.random.Generator
  ) = {
    val varianceNoSplit =
      targetAtSubset.sampleVariance * (targetAtSubset.length - 1d) / targetAtSubset.length

    var low = numConstant
    var high = attributes.length
    val N = attributes.length
    def swap(i: Int, j: Int) = {
      val t = attributes(i)
      attributes(i) = attributes(j)
      attributes(j) = t
    }

    var bestScore = Double.NegativeInfinity
    var bestFeature = -1
    var bestCutpoint = Double.NaN
    var visited = 0
    while (N - high < k && high - low > 0) {
      val r = rng.nextInt(low, high - 1)
      val attr = attributes(r)
      val (min, max) = minmax(takeCol(data, subset, attr))
      if (max <= min) {
        swap(r, low)
        low += 1
      } else {
        val cutpoint = rng.nextDouble(min, max)

        val take = takeCol(data, subset, attr) < cutpoint
        visited += 1
        val score = computeVarianceReduction(
          targetAtSubset,
          take,
          varianceNoSplit
        )

        if (score > bestScore) {
          bestScore = score
          bestFeature = attr
          bestCutpoint = cutpoint
        }
        swap(r, high - 1)
        high -= 1
      }
    }
    if (visited == 0) (-1, bestCutpoint, low)
    else
      (bestFeature, bestCutpoint, low)
  }

  def predictClassification(
      root: ClassificationTree,
      sample: Vec[Double]
  ): Vec[Double] = {
    def traverse(root: ClassificationTree): Vec[Double] = root match {
      case ClassificationLeaf(targetDistribution) => targetDistribution.toVec
      case ClassificationNonLeaf(left, right, splitFeature, cutpoint) =>
        if (sample.raw(splitFeature) < cutpoint) traverse(left)
        else traverse(right)
    }

    traverse(root)
  }

  def predictClassification(
      trees: Seq[ClassificationTree],
      samples: Mat[Double]
  ): Mat[Double] = {
    Mat(samples.rows.map { sample =>
      val preditionsOfTrees = trees.map(t => predictClassification(t, sample))

      Mat(preditionsOfTrees: _*).reduceRows((row, _) => row.mean2)
    }: _*).T
  }
  def predictRegression(
      root: RegressionTree,
      sample: Vec[Double]
  ): Double = {
    def traverse(root: RegressionTree): Double = root match {
      case RegressionLeaf(mean) => mean
      case RegressionNonLeaf(left, right, splitFeature, cutpoint) =>
        if (sample.raw(splitFeature) < cutpoint) traverse(left)
        else traverse(right)
    }

    traverse(root)
  }

  def predictRegression(
      trees: Seq[RegressionTree],
      samples: Mat[Double]
  ): Vec[Double] = {
    samples.rows.map { sample =>
      val preditionsOfTrees = trees.map(t => predictRegression(t, sample))

      preditionsOfTrees.toVec.mean2
    }.toVec
  }

  def buildForestClassification(
      data: Mat[Double],
      target: Vec[Int],
      sampleWeights: Option[Vec[Double]],
      numClasses: Int,
      nMin: Int,
      k: Int,
      m: Int,
      parallelism: Int,
      seed: Long = java.time.Instant.now.toEpochMilli
  ): Seq[ClassificationTree] = {
    val rng = org.saddle.spire.random.rng.Cmwc5.fromTime(seed)
    val subset = array.range(0, data.numRows).toVec
    sampleWeights.foreach(v =>
      require(!v.exists(_ < 0d), "Negative weights not allowed.")
    )
    val trees = if (parallelism <= 1) {
      0 until m map (_ =>
        buildTreeClassification(
          data,
          subset,
          target,
          sampleWeights,
          nMin,
          k,
          rng,
          numClasses,
          array.range(0, data.numCols),
          0
        )
      )
    } else {
      val fjp = new ForkJoinPool(parallelism)
      val ec = ExecutionContext.fromExecutorService(fjp)
      implicit val cs = IO.contextShift(ec)
      val trees = NonEmptyList
        .fromList(
          (0 until m).toList map (_ =>
            IO {
              buildTreeClassification(
                data,
                subset,
                target,
                sampleWeights,
                nMin,
                k,
                rng,
                numClasses,
                array.range(0, data.numCols),
                0
              )
            }
          )
        )
        .get
        .parSequence
        .unsafeRunSync
      fjp.shutdown()
      ec.shutdown()

      trees.toList
    }

    trees
  }

  def buildForestRegression(
      data: Mat[Double],
      target: Vec[Double],
      nMin: Int,
      k: Int,
      m: Int,
      parallelism: Int,
      seed: Long = java.time.Instant.now.toEpochMilli
  ): Seq[RegressionTree] = {
    val subset = array.range(0, data.numRows).toVec
    val rng = org.saddle.spire.random.rng.Cmwc5.fromTime(seed)
    val trees = if (parallelism <= 1) {
      0 until m map (_ =>
        buildTreeRegression(
          data,
          subset,
          target,
          nMin,
          k,
          rng,
          array.range(0, data.numCols),
          0
        )
      )
    } else {
      val fjp = new ForkJoinPool(parallelism)
      val ec = ExecutionContext.fromExecutorService(fjp)
      implicit val cs = IO.contextShift(ec)
      val trees = NonEmptyList
        .fromList(
          (0 until m).toList map (_ =>
            IO {
              buildTreeRegression(
                data,
                subset,
                target,
                nMin,
                k,
                rng,
                array.range(0, data.numCols),
                0
              )
            }
          )
        )
        .get
        .parSequence
        .unsafeRunSync
      fjp.shutdown()
      ec.shutdown()

      trees.toList
    }

    trees
  }

  def buildTreeRegression(
      data: Mat[Double],
      subset: Vec[Int],
      target: Vec[Double],
      nMin: Int,
      k: Int,
      rng: org.saddle.spire.random.Generator,
      attributes: Array[Int],
      numConstant: Int
  ): RegressionTree = {

    val targetInSubset = target.take(subset.toArray)
    def makeLeaf = {
      RegressionLeaf(targetInSubset.mean2)
    }
    def makeNonLeaf(
        leftTree: RegressionTree,
        rightTree: RegressionTree,
        splitFeatureIdx: Int,
        splitCutpoint: Double
    ) =
      RegressionNonLeaf(leftTree, rightTree, splitFeatureIdx, splitCutpoint)

    val targetIsConstant = {
      val col = targetInSubset
      val head = col.raw(0)
      var i = 1
      val n = col.length
      var uniform = true
      while (i < n && uniform) {
        if (col.raw(i) != head) {
          uniform = false
        }
        i += 1
      }
      uniform
    }
    if (subset.length < nMin) makeLeaf
    else if (targetIsConstant) makeLeaf
    else {

      val (splitFeatureIdx, splitCutpoint, nConstant2) =
        splitRegression(
          data,
          subset,
          attributes,
          numConstant,
          k,
          targetInSubset,
          rng
        )
      if (splitFeatureIdx == -1) makeLeaf
      else {

        val splitFeature = col(data, splitFeatureIdx)
        val leftSubset = subset.filter(s => splitFeature.raw(s) < splitCutpoint)
        val rightSubset =
          subset.filter(s => splitFeature.raw(s) >= splitCutpoint)
        val leftTree =
          buildTreeRegression(
            data,
            leftSubset,
            target,
            nMin,
            k,
            rng,
            attributes,
            nConstant2
          )
        val rightTree =
          buildTreeRegression(
            data,
            rightSubset,
            target,
            nMin,
            k,
            rng,
            attributes,
            nConstant2
          )
        makeNonLeaf(leftTree, rightTree, splitFeatureIdx, splitCutpoint)
      }
    }
  }

  def distribution(
      v: Vec[Int],
      sampleWeights: Option[Vec[Double]],
      numClasses: Int
  ) = {
    val ar = Array.ofDim[Double](numClasses)
    var i = 0
    val n = v.length
    if (sampleWeights.isEmpty) {
      val s = n.toDouble
      while (i < n) {
        val j = v.raw(i)
        ar(j) += 1d / s
        i += 1
      }
    } else {
      val w = sampleWeights.get
      var s = 0.0
      while (i < n) {
        val j = v.raw(i)
        val k = w.raw(i)
        ar(j) += k
        s += k
        i += 1
      }
      var j = 0
      while (j < numClasses) {
        ar(j) /= s
        j += 1
      }
    }
    ar.toVec
  }

  def takeCol(data: Mat[Double], rows: Vec[Int], col: Int): Vec[Double] = {
    data.toVec.slice(col, data.length, data.numCols).view(rows.toArray)
  }

  def col(data: Mat[Double], col: Int): Vec[Double] = {
    data.toVec.slice(col, data.length, data.numCols)
  }

  def buildTreeClassification(
      data: Mat[Double],
      subset: Vec[Int],
      target: Vec[Int],
      sampleWeights: Option[Vec[Double]],
      nMin: Int,
      k: Int,
      rng: org.saddle.spire.random.Generator,
      numClasses: Int,
      attributes: Array[Int],
      numConstant: Int
  ): ClassificationTree = {
    val targetInSubset = target.take(subset.toArray)
    val weightsInSubset = sampleWeights.map(w => w.take(subset.toArray))
    def makeLeaf = {
      val targetDistribution =
        distribution(targetInSubset, weightsInSubset, numClasses)
      ClassificationLeaf(targetDistribution.toSeq)
    }
    def makeNonLeaf(
        leftTree: ClassificationTree,
        rightTree: ClassificationTree,
        splitFeatureIdx: Int,
        splitCutpoint: Double
    ) =
      ClassificationNonLeaf(leftTree, rightTree, splitFeatureIdx, splitCutpoint)
    val targetIsConstant = {
      val col = targetInSubset
      val head = col.raw(0)
      var i = 1
      val n = col.length
      var uniform = true
      while (i < n && uniform) {
        if (col.raw(i) != head) {
          uniform = false
        }
        i += 1
      }
      uniform
    }
    if (data.numRows < nMin) makeLeaf
    else if (targetIsConstant) makeLeaf
    else {

      val (splitFeatureIdx, splitCutpoint, numConstant2) =
        splitClassification(
          data,
          subset,
          attributes,
          numConstant,
          k,
          targetInSubset,
          weightsInSubset,
          rng,
          numClasses
        )
      if (splitFeatureIdx < 0) makeLeaf
      else {
        val splitFeature = col(data, splitFeatureIdx)
        val leftSubset =
          subset.filter(s => splitFeature.raw(s) < splitCutpoint)
        val rightSubset =
          subset.filter(s => splitFeature.raw(s) >= splitCutpoint)

        val leftTree =
          buildTreeClassification(
            data,
            leftSubset,
            target,
            sampleWeights,
            nMin,
            k,
            rng,
            numClasses,
            attributes,
            numConstant2
          )
        val rightTree =
          buildTreeClassification(
            data,
            rightSubset,
            target,
            sampleWeights,
            nMin,
            k,
            rng,
            numClasses,
            attributes,
            numConstant2
          )
        makeNonLeaf(leftTree, rightTree, splitFeatureIdx, splitCutpoint)
      }
    }
  }

  def partition[@specialized(Int, Double) T: ClassTag](
      vec: Vec[T]
  )(pred: Array[Boolean]): (Vec[T], Vec[T]) = {
    var i = 0
    val n = vec.length
    val m = n / 2 + 1
    val bufT = new Buffer(new Array[T](m), 0)
    val bufF = new Buffer(new Array[T](m), 0)
    while (i < n) {
      val v: T = vec.raw(i)
      if (pred(i)) bufT.+=(v)
      else bufF.+=(v)
      i += 1
    }
    (Vec(bufT.toArray), Vec(bufF.toArray))
  }

  def giniScore(
      target: Vec[Int],
      sampleWeights: Option[Vec[Double]],
      samplesInSplit: Vec[Boolean],
      giniImpurityNoSplit: Double,
      numClasses: Int,
      buf1: Array[Double],
      buf2: Array[Double]
  ) = {
    val numSamplesNoSplit =
      if (sampleWeights.isEmpty) samplesInSplit.length.toDouble
      else sampleWeights.get.sum2
    var i = 0
    var targetInCount = 0.0
    var targetOutCount = 0.0
    val n = target.length

    val distributionIn = buf1
    val distributionOut = buf2
    if (sampleWeights.isEmpty) {
      while (i < n) {
        val v: Int = target.raw(i)
        if (samplesInSplit.raw(i)) {
          targetInCount += 1
          distributionIn(v) += 1d
        } else {
          targetOutCount += 1
          distributionOut(v) += 1d
        }
        i += 1
      }
    } else {
      val weights = sampleWeights.get
      while (i < n) {
        val v: Int = target.raw(i)
        val ww = weights.raw(i)
        if (samplesInSplit.raw(i)) {
          targetInCount += ww
          distributionIn(v) += ww
        } else {
          targetOutCount += ww
          distributionOut(v) += ww
        }
        i += 1
      }
    }
    i = 0
    while (i < numClasses) {
      distributionIn(i) /= targetInCount
      distributionOut(i) /= targetOutCount
      i += 1
    }

    val gIn = giniImpurityFromDistribution(distributionIn)
    val gOut = giniImpurityFromDistribution(distributionOut)

    giniImpurityNoSplit - gIn * targetInCount / numSamplesNoSplit - gOut * targetOutCount / numSamplesNoSplit
  }

  def giniImpurity(
      target: Vec[Int],
      weights: Option[Vec[Double]],
      numClasses: Int
  ): Double = {
    val p = distribution(target, weights, numClasses)
    val p2 = p * p
    1d - p2.sum
  }
  def giniImpurityFromDistribution(
      distribution: Array[Double]
  ): Double = {
    var s = 0d
    val n = distribution.length
    var i = 0
    while (i < n) {
      val k = distribution(i)
      distribution(i) = 0d
      s += k * k
      i += 1
    }
    1d - s
  }

  def computeVarianceReduction(
      target: Vec[Double],
      samplesInSplit: Vec[Boolean],
      varianceNoSplit: Double
  ) = {
    val (targetInSplit, targetOutSplit) =
      partition(target)(samplesInSplit.toArray)
    val varianceInSplit =
      if (targetInSplit.length == 1) 0d
      else
        targetInSplit.sampleVariance * (targetInSplit.length - 1d) / (targetInSplit.length)
    val varianceOutSplit =
      if (targetOutSplit.length == 1) 0d
      else
        targetOutSplit.sampleVariance * (targetOutSplit.length - 1d) / (targetOutSplit.length)

    val numSamplesNoSplit =
      target.length.toDouble

    (varianceNoSplit -
      (targetInSplit.length.toDouble / numSamplesNoSplit.toDouble) * varianceInSplit -
      (targetOutSplit.length.toDouble / numSamplesNoSplit.toDouble) * varianceOutSplit) / varianceNoSplit
  }

}
