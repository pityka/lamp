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

  private[extratrees] def oneHot(v: Vec[Int], classes: Int): Mat[Double] = {
    val zeros = 0 until classes map (_ => vec.zeros(v.length)) toArray
    var i = 0
    val n = v.length
    while (i < n) {
      val level = v.raw(i)
      zeros(level)(i) = 1d
      i += 1
    }
    Mat(zeros: _*)
  }

  def buildForestClassification(
      data: Mat[Double],
      target: Vec[Int],
      numClasses: Int,
      nMin: Int,
      k: Int,
      m: Int,
      parallelism: Int,
      seed: Long = java.time.Instant.now.toEpochMilli
  ): Seq[ClassificationTree] = {
    val rng = org.saddle.spire.random.rng.Cmwc5.fromTime(seed)
    val subset = array.range(0, data.numRows).toVec
    val trees = if (parallelism <= 1) {
      0 until m map (_ =>
        buildTreeClassification(
          data,
          subset,
          target,
          nMin,
          k,
          rng,
          numClasses
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
                nMin,
                k,
                rng,
                numClasses
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
        buildTreeRegression(data, subset, target, nMin, k, rng)
      )
    } else {
      val fjp = new ForkJoinPool(parallelism)
      val ec = ExecutionContext.fromExecutorService(fjp)
      implicit val cs = IO.contextShift(ec)
      val trees = NonEmptyList
        .fromList(
          (0 until m).toList map (_ =>
            IO { buildTreeRegression(data, subset, target, nMin, k, rng) }
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
      rng: org.saddle.spire.random.Generator
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
      val shuffled =
        array.shuffle(array.range(0, data.numCols), rng)
      val candidateFeatures = shuffled.iterator
        .filter { colIdx =>
          val col = takeCol(data, subset, colIdx)
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
          !uniform
        }
        .take(k)
        .toArray

      if (candidateFeatures.isEmpty) makeLeaf
      else {
        val (splitFeatureIdx, splitCutpoint) =
          splitRegression(
            data,
            subset,
            candidateFeatures.toVec,
            targetInSubset,
            rng
          )
        val splitFeature = data.col(splitFeatureIdx)
        val leftSubset = subset.filter(s => splitFeature.raw(s) < splitCutpoint)
        val rightSubset =
          subset.filter(s => splitFeature.raw(s) >= splitCutpoint)
        val leftTree =
          buildTreeRegression(data, leftSubset, target, nMin, k, rng)
        val rightTree =
          buildTreeRegression(data, rightSubset, target, nMin, k, rng)
        makeNonLeaf(leftTree, rightTree, splitFeatureIdx, splitCutpoint)
      }
    }
  }

  def distribution(v: Vec[Int], numClasses: Int) = {
    val ar = Array.ofDim[Double](numClasses)
    var i = 0
    val n = v.length
    val s = n.toDouble
    while (i < n) {
      val j = v.raw(i)
      ar(j) += 1d / s
      i += 1
    }
    ar.toVec
  }

  def takeCol(data: Mat[Double], rows: Vec[Int], col: Int): Vec[Double] = {
    data.toVec.slice(col, data.length, data.numCols).view(rows.toArray)
  }

  def buildTreeClassification(
      data: Mat[Double],
      subset: Vec[Int],
      target: Vec[Int],
      nMin: Int,
      k: Int,
      rng: org.saddle.spire.random.Generator,
      numClasses: Int
  ): ClassificationTree = {
    val targetInSubset = target.take(subset.toArray)
    def makeLeaf = {
      val targetDistribution = distribution(targetInSubset, numClasses)
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
      val shuffled =
        array.shuffle(array.range(0, data.numCols), rng)
      val candidateFeatures = shuffled.iterator
        .filter { colIdx =>
          val col = takeCol(data, subset, colIdx)
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
          !uniform
        }
        .take(k)
        .toArray

      if (candidateFeatures.isEmpty) makeLeaf
      else {

        val (splitFeatureIdx, splitCutpoint) =
          splitClassification(
            data,
            subset,
            candidateFeatures.toVec,
            targetInSubset,
            rng,
            numClasses
          )
        val splitFeature = data.col(splitFeatureIdx)
        val leftSubset =
          subset.filter(s => splitFeature.raw(s) < splitCutpoint)
        val rightSubset =
          subset.filter(s => splitFeature.raw(s) >= splitCutpoint)

        val leftTree =
          buildTreeClassification(
            data,
            leftSubset,
            target,
            nMin,
            k,
            rng,
            numClasses
          )
        val rightTree =
          buildTreeClassification(
            data,
            rightSubset,
            target,
            nMin,
            k,
            rng,
            numClasses
          )
        makeNonLeaf(leftTree, rightTree, splitFeatureIdx, splitCutpoint)
      }
    }
  }

  def splitClassification(
      data: Mat[Double],
      subset: Vec[Int],
      attributes: Vec[Int],
      targetAtSubset: Vec[Int],
      rng: org.saddle.spire.random.Generator,
      numClasses: Int
  ) = {
    val min = attributes.map(i => takeCol(data, subset, i).min2)
    val max = attributes.map(i => takeCol(data, subset, i).max2)
    val cutpoints =
      min.zipMap(max)((min, max) => rng.nextDouble(from = min, until = max))
    val giniTotal = giniImpurity(targetAtSubset, numClasses)
    val scores = cutpoints
      .zipMapIdx { (cutpoint, colIdx) =>
        val c2 = attributes.raw(colIdx)
        val take = takeCol(data, subset, c2) < cutpoint

        giniScore(
          targetAtSubset,
          take,
          giniTotal,
          numClasses
        )
      }

    val sidx = scores.argmax
    val splitAttribute = attributes.raw(sidx)
    val splitCutpoint = cutpoints.raw(sidx)

    (splitAttribute, splitCutpoint)
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
      samplesInSplit: Vec[Boolean],
      giniImpurityNoSplit: Double,
      numClasses: Int
  ) = {
    val numSamplesNoSplit = samplesInSplit.length
    val (targetIn, targetOut) = partition(target)(samplesInSplit.toArray)
    val gIn = giniImpurity(targetIn, numClasses)
    val gOut =
      giniImpurity(targetOut, numClasses)
    giniImpurityNoSplit - gIn * targetIn.length / numSamplesNoSplit - gOut * targetOut.length / numSamplesNoSplit
  }
  def giniImpurity(
      target: Vec[Int],
      numClasses: Int
  ): Double = {
    val p = distribution(target, numClasses)
    val p2 = p * p
    1d - p2.sum
  }
  def splitRegression(
      data: Mat[Double],
      subset: Vec[Int],
      attributes: Vec[Int],
      target: Vec[Double],
      rng: org.saddle.spire.random.Generator
  ) = {
    val min = attributes.map(i => takeCol(data, subset, i).min2)
    val max = attributes.map(i => takeCol(data, subset, i).max2)
    val cutpoints =
      min.zipMap(max)((min, max) => rng.nextDouble(from = min, until = max))
    val varianceNoSplit =
      target.sampleVariance * (target.length - 1d) / target.length
    val scores = cutpoints
      .zipMapIdx { (cutpoint, colIdx) =>
        val c2 = attributes.raw(colIdx)
        val take = takeCol(data, subset, c2) < cutpoint

        val score = computeVarianceReduction(
          target,
          take,
          varianceNoSplit
        )

        score
      }

    val sidx = scores.argmax
    val splitAttribute = attributes.raw(sidx)
    val splitCutpoint = cutpoints.raw(sidx)

    (splitAttribute, splitCutpoint)
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
