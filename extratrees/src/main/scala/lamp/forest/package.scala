package lamp

import org.saddle._
import org.saddle.macros.BinOps._
import java.util.concurrent.ForkJoinPool
import scala.concurrent.ExecutionContext
import cats.effect.IO
import cats.syntax.parallel._
import cats.data.NonEmptyList

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
      parallelism: Int
  ): Seq[ClassificationTree] = {
    val oh = oneHot(target, numClasses)
    val subset = array.range(0, data.numRows).toVec
    val trees = if (parallelism <= 1) {
      0 until m map (_ => buildTreeClassification(data, subset, oh, nMin, k))
    } else {
      val fjp = new ForkJoinPool(parallelism)
      val ec = ExecutionContext.fromExecutorService(fjp)
      implicit val cs = IO.contextShift(ec)
      val trees = NonEmptyList
        .fromList(
          (0 until m).toList map (_ =>
            IO { buildTreeClassification(data, subset, oh, nMin, k) }
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
      parallelism: Int
  ): Seq[RegressionTree] = {
    val subset = array.range(0, data.numRows).toVec

    val trees = if (parallelism <= 1) {
      0 until m map (_ => buildTreeRegression(data, subset, target, nMin, k))
    } else {
      val fjp = new ForkJoinPool(parallelism)
      val ec = ExecutionContext.fromExecutorService(fjp)
      implicit val cs = IO.contextShift(ec)
      val trees = NonEmptyList
        .fromList(
          (0 until m).toList map (_ =>
            IO { buildTreeRegression(data, subset, target, nMin, k) }
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
      k: Int
  ): RegressionTree = {

    val rng = org.saddle.spire.random.rng.Cmwc5.fromTime()
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
      targetInSubset.sampleVariance == 0d
    }
    if (subset.length < nMin) makeLeaf
    else if (targetIsConstant) makeLeaf
    else {
      val nonConstantFeatures = data
        .row(subset.toArray)
        .reduceCols { (col, _) =>
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
        .find(uniform => !uniform)
      if (nonConstantFeatures.isEmpty) makeLeaf
      else {
        val candidateFeatures =
          array.shuffle(nonConstantFeatures.toArray, rng).take(k)
        val (splitFeatureIdx, splitCutpoint) =
          splitRegression(data, subset, candidateFeatures.toVec, target, rng)
        val splitFeature = data.col(splitFeatureIdx)
        val leftSubset = subset.filter(s => splitFeature.raw(s) < splitCutpoint)
        val rightSubset =
          subset.filter(s => splitFeature.raw(s) >= splitCutpoint)
        val leftTree =
          buildTreeRegression(data, leftSubset, target, nMin, k)
        val rightTree =
          buildTreeRegression(data, rightSubset, target, nMin, k)
        makeNonLeaf(leftTree, rightTree, splitFeatureIdx, splitCutpoint)
      }
    }
  }

  def buildTreeClassification(
      data: Mat[Double],
      subset: Vec[Int],
      target: Mat[Double],
      nMin: Int,
      k: Int
  ): ClassificationTree = {
    val rng = org.saddle.spire.random.rng.Cmwc5.fromTime()
    val targetInSubset = target.row(subset.toArray)
    def makeLeaf = {
      val targetDistribution = targetInSubset.reduceCols((col, _) => col.mean2)
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
      targetInSubset.reduceCols((col, _) => col.sum2).countif(_ > 0d) == 1
    }
    if (subset.length < nMin) makeLeaf
    else if (targetIsConstant) makeLeaf
    else {
      val nonConstantFeatures = data
        .row(subset.toArray)
        .reduceCols { (col, _) =>
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
        .find(uniform => !uniform)
      if (nonConstantFeatures.isEmpty) makeLeaf
      else {
        val candidateFeatures =
          array.shuffle(nonConstantFeatures.toArray, rng).take(k)
        val (splitFeatureIdx, splitCutpoint) =
          splitClassification(
            data,
            subset,
            candidateFeatures.toVec,
            target,
            rng
          )
        val splitFeature = data.col(splitFeatureIdx)
        val leftSubset = subset.filter(s => splitFeature.raw(s) < splitCutpoint)
        val rightSubset =
          subset.filter(s => splitFeature.raw(s) >= splitCutpoint)
        val leftTree =
          buildTreeClassification(data, leftSubset, target, nMin, k)
        val rightTree =
          buildTreeClassification(data, rightSubset, target, nMin, k)
        makeNonLeaf(leftTree, rightTree, splitFeatureIdx, splitCutpoint)
      }
    }
  }

  def splitClassification(
      data: Mat[Double],
      subset: Vec[Int],
      attributes: Vec[Int],
      target: Mat[Double],
      rng: org.saddle.spire.random.Generator
  ) = {
    val data1 =
      if (subset.length < data.numRows) data.row(subset.toArray) else data
    val min = attributes.map(i => data1.col(i).min2)
    val max = attributes.map(i => data1.col(i).max2)
    val cutpoints =
      min.zipMap(max)((min, max) => rng.nextDouble(from = min, until = max))
    val giniTotal = giniImpurity(subset, target)
    val scores = cutpoints
      .zipMapIdx { (cutpoint, colIdx) =>
        val c2 = attributes.raw(colIdx)
        val take = data1.col(c2) < cutpoint
        val samplesInSplit = subset.where(take)
        val samplesOutSplit = subset.where(take.map(b => !b))

        giniScore(target, samplesInSplit, samplesOutSplit, giniTotal)
      }

    val sidx = scores.argmax
    val splitAttribute = attributes.raw(sidx)
    val splitCutpoint = cutpoints.raw(sidx)

    (splitAttribute, splitCutpoint)
  }

  def giniScore(
      target: Mat[Double],
      samplesInSplit: Vec[Int],
      samplesOutSplit: Vec[Int],
      giniImpurityNoSplit: Double
  ) = {
    val numSamplesNoSplit =
      samplesInSplit.length + samplesOutSplit.length.toDouble
    val gIn = giniImpurity(samplesInSplit, target)
    val gOut = giniImpurity(samplesOutSplit, target)
    giniImpurityNoSplit - gIn * samplesInSplit.length / numSamplesNoSplit - gOut * samplesOutSplit.length / numSamplesNoSplit
  }
  def giniImpurity(samplesInSplit: Vec[Int], target: Mat[Double]) = {
    val targetInSplit = target.row(samplesInSplit.toArray)
    val p = targetInSplit.reduceCols((col, _) => col.mean2)
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
    val data1 =
      if (subset.length < data.numRows) data.row(subset.toArray) else data
    val min = attributes.map(i => data1.col(i).min2)
    val max = attributes.map(i => data1.col(i).max2)

    val cutpoints =
      min.zipMap(max)((min, max) => rng.nextDouble(from = min, until = max))
    val targetNoSplit = target.take(subset.toArray)
    val varianceNoSplit =
      targetNoSplit.sampleVariance * (targetNoSplit.length - 1d) / targetNoSplit.length
    val scores = cutpoints
      .zipMapIdx { (cutpoint, colIdx) =>
        val c2 = attributes.raw(colIdx)
        val take = data1.col(c2) < cutpoint
        val samplesInSplit = subset.where(take)
        val samplesOutSplit = subset.where(take.map(b => !b))
        val score = computeVarianceReduction(
          target,
          samplesInSplit,
          samplesOutSplit,
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
      samplesInSplit: Vec[Int],
      samplesOutSplit: Vec[Int],
      varianceNoSplit: Double
  ) = {
    val targetInSplit = target.take(samplesInSplit.toArray)
    val targetOutSplit = target.take(samplesOutSplit.toArray)
    val varianceInSplit =
      if (targetInSplit.length == 1) 0d
      else
        targetInSplit.sampleVariance * (targetInSplit.length - 1d) / (targetInSplit.length)
    val varianceOutSplit =
      if (targetOutSplit.length == 1) 0d
      else
        targetOutSplit.sampleVariance * (targetOutSplit.length - 1d) / (targetOutSplit.length)

    val numSamplesNoSplit =
      samplesInSplit.length + samplesOutSplit.length.toDouble

    (varianceNoSplit -
      (samplesInSplit.length.toDouble / numSamplesNoSplit.toDouble) * varianceInSplit -
      (samplesOutSplit.length.toDouble / numSamplesNoSplit.toDouble) * varianceOutSplit) / varianceNoSplit
  }

}
