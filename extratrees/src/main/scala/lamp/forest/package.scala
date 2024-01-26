package lamp

import org.saddle._
import cats.effect.IO
import cats.effect.syntax.all._
import cats.effect.unsafe.implicits.global

package object extratrees {

  private[lamp] def lessThanCutpoint(
      v: Vec[Double],
      cutpoint: Double,
      missingIsLess: Boolean
  ) = {
    val l = v.length
    val ar = new Array[Boolean](l)
    var i = 0
    if (missingIsLess) {
      while (i < l) {
        val x = v.raw(i)
        ar(i) = x.isNaN || x < cutpoint
        i += 1
      }
    } else {
      while (i < l) {
        val x = v.raw(i)
        ar(i) = x < cutpoint
        i += 1
      }
    }
    Vec(ar)
  }

  private[lamp] def minmax(self: Vec[Double]) = {
    var sMin = Double.MaxValue
    var sMax = Double.MinValue
    var hasMissing = false
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
      if (v.isNaN && !hasMissing) {
        hasMissing = true
      }
      i += 1
    }
    (sMin, sMax, hasMissing)
  }

  private[lamp] def splitBestClassification(
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
    var bestMissingIsLess = false
    var visited = 0
    while (N - high < k && high - low > 0) {
      val r = rng.nextInt(low, high - 1)
      val attr = attributes(r)
      val (min, max, hasMissing) = minmax(takeCol(data, subset, attr))
      if (max <= min && !hasMissing) {
        swap(r, low)
        low += 1
      } else {
        val col = takeCol(data, subset, attr)
        val cutpoint = {
          def score(candidateCutPoint: Int): Double = {
            val cutpoint = col.raw(candidateCutPoint)
            val takeMissingIsLess =
              if (!hasMissing) null
              else
                lessThanCutpoint(col, cutpoint, true)
            val takeMissingIsNotLess =
              lessThanCutpoint(col, cutpoint, false)

            val scoreMissingIsLess =
              if (!hasMissing) Double.NaN
              else
                giniScore(
                  targetAtSubset,
                  weightsAtSubset,
                  takeMissingIsLess,
                  giniTotal,
                  numClasses,
                  buf1,
                  buf2
                )
            val scoreMissingIsNotLess = giniScore(
              targetAtSubset,
              weightsAtSubset,
              takeMissingIsNotLess,
              giniTotal,
              numClasses,
              buf1,
              buf2
            )

            val chosenMissingIsLess =
              (!scoreMissingIsLess.isNaN && (scoreMissingIsLess > scoreMissingIsNotLess || scoreMissingIsNotLess.isNaN))
            val chosenScore =
              if (chosenMissingIsLess) scoreMissingIsLess
              else scoreMissingIsNotLess
            chosenScore
          }
          var i = 0
          val n = col.length
          var max = Double.NegativeInfinity
          var maxi = 0
          while (i < n) {
            val s = score(i)
            if (s > max) {
              max = s
              maxi = i
            }
            i += 1
          }
          col(maxi)
        }

        val takeMissingIsLess =
          if (!hasMissing) null
          else
            lessThanCutpoint(col, cutpoint, true)
        val takeMissingIsNotLess =
          lessThanCutpoint(col, cutpoint, false)

        val scoreMissingIsLess =
          if (!hasMissing) Double.NaN
          else
            giniScore(
              targetAtSubset,
              weightsAtSubset,
              takeMissingIsLess,
              giniTotal,
              numClasses,
              buf1,
              buf2
            )
        val scoreMissingIsNotLess = giniScore(
          targetAtSubset,
          weightsAtSubset,
          takeMissingIsNotLess,
          giniTotal,
          numClasses,
          buf1,
          buf2
        )

        val chosenMissingIsLess =
          (!scoreMissingIsLess.isNaN && (scoreMissingIsLess > scoreMissingIsNotLess || scoreMissingIsNotLess.isNaN))
        val chosenScore =
          if (chosenMissingIsLess) scoreMissingIsLess else scoreMissingIsNotLess

        if (chosenScore > bestScore) {
          bestScore = chosenScore
          bestFeature = attr
          bestCutpoint = cutpoint
          bestMissingIsLess = chosenMissingIsLess
        }
        if (chosenScore.isNaN) {
          swap(r, low)
          low += 1
        } else {
          visited += 1
          swap(r, high - 1)
          high -= 1
        }
      }
    }
    if (visited == 0 || bestCutpoint.isNaN)
      (-1, bestCutpoint, low, bestMissingIsLess)
    else
      (bestFeature, bestCutpoint, low, bestMissingIsLess)
  }
  private[lamp] def splitClassification(
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
    var bestMissingIsLess = false
    var visited = 0
    while (N - high < k && high - low > 0) {
      val r = rng.nextInt(low, high - 1)
      val attr = attributes(r)
      val (min, max, hasMissing) = minmax(takeCol(data, subset, attr))
      if (max <= min && !hasMissing) {
        swap(r, low)
        low += 1
      } else {
        val cutpoint = rng.nextDouble(min, max)

        val col = takeCol(data, subset, attr)
        val takeMissingIsLess =
          if (!hasMissing) null
          else
            lessThanCutpoint(col, cutpoint, true)
        val takeMissingIsNotLess =
          lessThanCutpoint(col, cutpoint, false)

        val scoreMissingIsLess =
          if (!hasMissing) Double.NaN
          else
            giniScore(
              targetAtSubset,
              weightsAtSubset,
              takeMissingIsLess,
              giniTotal,
              numClasses,
              buf1,
              buf2
            )
        val scoreMissingIsNotLess = giniScore(
          targetAtSubset,
          weightsAtSubset,
          takeMissingIsNotLess,
          giniTotal,
          numClasses,
          buf1,
          buf2
        )

        val chosenMissingIsLess =
          (!scoreMissingIsLess.isNaN && (scoreMissingIsLess > scoreMissingIsNotLess || scoreMissingIsNotLess.isNaN))
        val chosenScore =
          if (chosenMissingIsLess) scoreMissingIsLess else scoreMissingIsNotLess

        if (chosenScore > bestScore) {
          bestScore = chosenScore
          bestFeature = attr
          bestCutpoint = cutpoint
          bestMissingIsLess = chosenMissingIsLess
        }
        if (chosenScore.isNaN) {
          swap(r, low)
          low += 1
        } else {
          visited += 1
          swap(r, high - 1)
          high -= 1
        }
      }
    }
    if (visited == 0 || bestCutpoint.isNaN)
      (-1, bestCutpoint, low, bestMissingIsLess)
    else
      (bestFeature, bestCutpoint, low, bestMissingIsLess)
  }
  private[lamp] def splitBestRegression(
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
    var bestMissingIsLess = false
    var visited = 0
    while (N - high < k && high - low > 0) {
      val r = rng.nextInt(low, high - 1)
      val attr = attributes(r)
      val (min, max, hasMissing) = minmax(takeCol(data, subset, attr))
      if (max <= min && !hasMissing) {
        swap(r, low)
        low += 1
      } else {
        val col = takeCol(data, subset, attr)
        val cutpoint = {
          def score(candidateCutPoint: Int): Double = {
            val cutpoint = col.raw(candidateCutPoint)
            val takeMissingIsLess =
              if (!hasMissing) null
              else
                lessThanCutpoint(col, cutpoint, true)
            val takeMissingIsNotLess =
              lessThanCutpoint(col, cutpoint, false)

            val scoreMissingIsLess =
              if (!hasMissing) Double.NaN
              else
                computeVarianceReduction(
                  targetAtSubset,
                  takeMissingIsLess,
                  varianceNoSplit
                )
            val scoreMissingIsNotLess = computeVarianceReduction(
              targetAtSubset,
              takeMissingIsNotLess,
              varianceNoSplit
            )

            val chosenMissingIsLess =
              (!scoreMissingIsLess.isNaN && (scoreMissingIsLess > scoreMissingIsNotLess || scoreMissingIsNotLess.isNaN))
            val chosenScore =
              if (chosenMissingIsLess) scoreMissingIsLess
              else scoreMissingIsNotLess
            chosenScore
          }
          var i = 0
          val n = col.length
          var max = Double.NegativeInfinity
          var maxi = 0
          while (i < n) {
            val s = score(i)
            if (s > max) {
              max = s
              maxi = i
            }
            i += 1
          }
          col(maxi)
        }
        val takeMissingIsLess =
          if (!hasMissing) null
          else
            lessThanCutpoint(col, cutpoint, true)
        val takeMissingIsNotLess =
          lessThanCutpoint(col, cutpoint, false)

        val scoreMissingIsLess =
          if (!hasMissing) Double.NaN
          else
            computeVarianceReduction(
              targetAtSubset,
              takeMissingIsLess,
              varianceNoSplit
            )
        val scoreMissingIsNotLess = computeVarianceReduction(
          targetAtSubset,
          takeMissingIsNotLess,
          varianceNoSplit
        )

        val chosenMissingIsLess =
          (!scoreMissingIsLess.isNaN && (scoreMissingIsLess > scoreMissingIsNotLess || scoreMissingIsNotLess.isNaN))
        val chosenScore =
          if (chosenMissingIsLess) scoreMissingIsLess else scoreMissingIsNotLess

        if (chosenScore > bestScore) {
          bestScore = chosenScore
          bestFeature = attr
          bestCutpoint = cutpoint
          bestMissingIsLess = chosenMissingIsLess
        }
        if (chosenScore.isNaN) {
          swap(r, low)
          low += 1
        } else {
          visited += 1
          swap(r, high - 1)
          high -= 1
        }

      }
    }
    if (visited == 0 || bestCutpoint.isNaN)
      (-1, bestCutpoint, low, bestMissingIsLess)
    else
      (bestFeature, bestCutpoint, low, bestMissingIsLess)

  }
  private[lamp] def splitRegression(
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
    var bestMissingIsLess = false
    var visited = 0
    while (N - high < k && high - low > 0) {
      val r = rng.nextInt(low, high - 1)
      val attr = attributes(r)
      val (min, max, hasMissing) = minmax(takeCol(data, subset, attr))
      if (max <= min && !hasMissing) {
        swap(r, low)
        low += 1
      } else {
        val cutpoint = rng.nextDouble(min, max)
        val col = takeCol(data, subset, attr)
        val takeMissingIsLess =
          if (!hasMissing) null
          else
            lessThanCutpoint(col, cutpoint, true)
        val takeMissingIsNotLess =
          lessThanCutpoint(col, cutpoint, false)

        val scoreMissingIsLess =
          if (!hasMissing) Double.NaN
          else
            computeVarianceReduction(
              targetAtSubset,
              takeMissingIsLess,
              varianceNoSplit
            )
        val scoreMissingIsNotLess = computeVarianceReduction(
          targetAtSubset,
          takeMissingIsNotLess,
          varianceNoSplit
        )

        val chosenMissingIsLess =
          (!scoreMissingIsLess.isNaN && (scoreMissingIsLess > scoreMissingIsNotLess || scoreMissingIsNotLess.isNaN))
        val chosenScore =
          if (chosenMissingIsLess) scoreMissingIsLess else scoreMissingIsNotLess

        if (chosenScore > bestScore) {
          bestScore = chosenScore
          bestFeature = attr
          bestCutpoint = cutpoint
          bestMissingIsLess = chosenMissingIsLess
        }
        if (chosenScore.isNaN) {
          swap(r, low)
          low += 1
        } else {
          visited += 1
          swap(r, high - 1)
          high -= 1
        }

      }
    }
    if (visited == 0 || bestCutpoint.isNaN)
      (-1, bestCutpoint, low, bestMissingIsLess)
    else
      (bestFeature, bestCutpoint, low, bestMissingIsLess)

  }

  private[lamp] def predictClassification(
      root: ClassificationTree,
      sample: Vec[Double]
  ): Vec[Double] = {
    def traverse(root: ClassificationTree): Vec[Double] = root match {
      case ClassificationLeaf(targetDistribution) => targetDistribution.toVec
      case ClassificationNonLeaf(
            left,
            right,
            splitFeature,
            cutpoint,
            missingIsLess
          ) =>
        if (
          sample.raw(splitFeature) < cutpoint || (missingIsLess && sample
            .raw(splitFeature)
            .isNaN)
        ) traverse(left)
        else traverse(right)
    }

    traverse(root)
  }

  /** Prediction from a set of trees
    *
    * Returns a matrix of nxm where n is the number of samples m is the number
    * of classes, column c corresponds to class c.
    */
  def predictClassification(
      trees: Seq[ClassificationTree],
      samples: Mat[Double]
  ): Mat[Double] = {
    Mat(samples.rows.map { sample =>
      val preditionsOfTrees = trees.map(t => predictClassification(t, sample))

      Mat(preditionsOfTrees: _*).reduceRows((row, _) => row.mean2)
    }: _*).T
  }

  private[lamp] def predictRegression(
      root: RegressionTree,
      sample: Vec[Double]
  ): Double = {
    def traverse(root: RegressionTree): Double = root match {
      case RegressionLeaf(mean) => mean
      case RegressionNonLeaf(
            left,
            right,
            splitFeature,
            cutpoint,
            missingIsLess
          ) =>
        if (
          sample.raw(splitFeature) < cutpoint || (missingIsLess && sample
            .raw(splitFeature)
            .isNaN)
        ) traverse(left)
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

  /** Train an extratrees classifier forest
    *
    * @param data
    * @param target
    * @param sampleWeights
    * @param numClasses
    * @param nMin
    *   minimum sample size for splitting a node
    * @param k
    *   number of features to consider in each split step. The best among these
    *   will be chosen.
    * @param m
    *   number of trees
    * @param parallelism
    * @param bestSplit
    *   if true then the split is not random but the best among possible splits.
    * @param maxDepth
    *   maximum tree depth
    * @param seed
    *
    * Returns a list of ClassificationTree objects which can be passed to
    * `predictClassification`
    */
  def buildForestClassification(
      data: Mat[Double],
      target: Vec[Int],
      sampleWeights: Option[Vec[Double]],
      numClasses: Int,
      nMin: Int,
      k: Int,
      m: Int,
      parallelism: Int,
      bestSplit: Boolean = false,
      maxDepth: Int = Int.MaxValue,
      seed: Long = java.time.Instant.now.toEpochMilli
  ): Seq[ClassificationTree] = {
    require(
      data.numRows == target.length,
      s"Data.numRows(${data.numRows}) != target.length (${target.length})"
    )

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
          0,
          bestSplit,
          maxDepth,
          0
        )
      )
    } else {
      val trees =
        (0 until m).toList
          .map(_ => org.saddle.spire.random.rng.Cmwc5.fromTime(rng.nextLong()))
          .parTraverseN(parallelism)(rng =>
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
                0,
                bestSplit,
                maxDepth,
                0
              )
            }
          )
          .unsafeRunSync()

      trees.toList
    }

    trees
  }

  /** Train an extratrees regression forest
    *
    * @param data
    * @param target
    * @param nMin
    *   minimum sample size for splitting a node
    * @param k
    *   number of features to consider in each split step. The best among these
    *   will be chosen.
    * @param m
    *   number of trees
    * @param parallelism
    * @param bestSplit
    *   if true then the split is not random but the best among possible splits.
    * @param maxDepth
    *   maximum tree depth
    * @param seed
    *
    * Returns a list of RegressionTree objects which can be passed to
    * `predictRegression`
    */
  def buildForestRegression(
      data: Mat[Double],
      target: Vec[Double],
      nMin: Int,
      k: Int,
      m: Int,
      parallelism: Int,
      bestSplit: Boolean = false,
      maxDepth: Int = Int.MaxValue,
      seed: Long = java.time.Instant.now.toEpochMilli
  ): Seq[RegressionTree] = {
    require(
      data.numRows == target.length,
      s"Data.numRows(${data.numRows}) != target.length (${target.length})"
    )
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
          0,
          bestSplit,
          maxDepth,
          0
        )
      )
    } else {
      val trees =
        (0 until m).toList
          .map(_ => org.saddle.spire.random.rng.Cmwc5.fromTime(rng.nextLong()))
          .parTraverseN(parallelism)(rng =>
            IO {
              buildTreeRegression(
                data,
                subset,
                target,
                nMin,
                k,
                rng,
                array.range(0, data.numCols),
                0,
                bestSplit,
                maxDepth,
                0
              )
            }
          )
          .unsafeRunSync()

      trees.toList
    }

    trees
  }

  private[lamp] def buildTreeRegression(
      data: Mat[Double],
      subset: Vec[Int],
      target: Vec[Double],
      nMin: Int,
      k: Int,
      rng: org.saddle.spire.random.Generator,
      attributes: Array[Int],
      numConstant: Int,
      bestSplit: Boolean,
      maxDepth: Int,
      currentDepth: Int
  ): RegressionTree = {
    require(subset.length > 0)
    val targetInSubset = target.take(subset.toArray)
    def makeLeaf = {
      RegressionLeaf(targetInSubset.mean2)
    }
    def makeNonLeaf(
        leftTree: RegressionTree,
        rightTree: RegressionTree,
        splitFeatureIdx: Int,
        splitCutpoint: Double,
        splitMissingIsLess: Boolean
    ) =
      RegressionNonLeaf(
        leftTree,
        rightTree,
        splitFeatureIdx,
        splitCutpoint,
        splitMissingIsLess
      )

    def targetIsConstant = {
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
    if (subset.length < nMin || currentDepth >= maxDepth) makeLeaf
    else if (targetIsConstant) makeLeaf
    else {

      val (splitFeatureIdx, splitCutpoint, nConstant2, missingIsLess) =
        if (bestSplit)
          splitBestRegression(
            data,
            subset,
            attributes,
            numConstant,
            k,
            targetInSubset,
            rng
          )
        else
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

        val leftSubset =
          if (missingIsLess)
            subset.filter(s =>
              splitFeature.raw(s) < splitCutpoint || splitFeature.raw(s).isNaN
            )
          else subset.filter(s => splitFeature.raw(s) < splitCutpoint)

        val rightSubset =
          if (missingIsLess)
            subset.filter(s => splitFeature.raw(s) >= splitCutpoint)
          else
            subset.filter(s =>
              splitFeature.raw(s) >= splitCutpoint || splitFeature.raw(s).isNaN
            )

        val leftTree =
          buildTreeRegression(
            data,
            leftSubset,
            target,
            nMin,
            k,
            rng,
            attributes,
            nConstant2,
            bestSplit,
            maxDepth,
            currentDepth + 1
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
            nConstant2,
            bestSplit,
            maxDepth,
            currentDepth
          )
        makeNonLeaf(
          leftTree,
          rightTree,
          splitFeatureIdx,
          splitCutpoint,
          missingIsLess
        )
      }
    }
  }

  private[lamp] def distribution(
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

  private[lamp] def takeCol(
      data: Mat[Double],
      rows: Vec[Int],
      col: Int
  ): Vec[Double] = {
    data.toVec.slice(col, data.length, data.numCols).view(rows.toArray)
  }

  private[lamp] def col(data: Mat[Double], col: Int): Vec[Double] = {
    data.toVec.slice(col, data.length, data.numCols)
  }

  private[lamp] def buildTreeClassification(
      data: Mat[Double],
      subset: Vec[Int],
      target: Vec[Int],
      sampleWeights: Option[Vec[Double]],
      nMin: Int,
      k: Int,
      rng: org.saddle.spire.random.Generator,
      numClasses: Int,
      attributes: Array[Int],
      numConstant: Int,
      bestSplit: Boolean,
      maxDepth: Int,
      currentDepth: Int
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
        splitCutpoint: Double,
        splitMissingIsLess: Boolean
    ) =
      ClassificationNonLeaf(
        leftTree,
        rightTree,
        splitFeatureIdx,
        splitCutpoint,
        splitMissingIsLess
      )
    def targetIsConstant = {
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
    if (data.numRows < nMin || currentDepth >= maxDepth) makeLeaf
    else if (targetIsConstant) makeLeaf
    else {

      val (splitFeatureIdx, splitCutpoint, numConstant2, missingIsLess) =
        if (bestSplit)
          splitBestClassification(
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
        else
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
          if (missingIsLess)
            subset.filter(s =>
              splitFeature.raw(s) < splitCutpoint || splitFeature.raw(s).isNaN
            )
          else subset.filter(s => splitFeature.raw(s) < splitCutpoint)

        val rightSubset =
          if (missingIsLess)
            subset.filter(s => splitFeature.raw(s) >= splitCutpoint)
          else
            subset.filter(s =>
              splitFeature.raw(s) >= splitCutpoint || splitFeature.raw(s).isNaN
            )

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
            numConstant2,
            bestSplit,
            maxDepth,
            currentDepth + 1
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
            numConstant2,
            bestSplit,
            maxDepth,
            currentDepth + 1
          )
        makeNonLeaf(
          leftTree,
          rightTree,
          splitFeatureIdx,
          splitCutpoint,
          missingIsLess
        )
      }
    }
  }

  private[lamp] def partition[@specialized(Int, Double) T: ST](
      vec: Vec[T]
  )(pred: Array[Boolean]): (Vec[T], Vec[T]) = {
    var i = 0
    val n = vec.length
    val m = n / 2 + 1
    val bufT = Buffer.empty[T](m)
    val bufF = Buffer.empty[T](m)
    while (i < n) {
      val v: T = vec.raw(i)
      if (pred(i)) bufT.+=(v)
      else bufF.+=(v)
      i += 1
    }
    (Vec(bufT.toArray), Vec(bufF.toArray))
  }

  private[lamp] def giniScore(
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

  private[lamp] def giniImpurity(
      target: Vec[Int],
      weights: Option[Vec[Double]],
      numClasses: Int
  ): Double = {

    def sqSum(a: Array[Double]): Double = {
      var i = 0
      var s = 0d
      val l = a.length
      while (i < l) {
        val x = a(i)
        s += x * x
        i += 1
      }
      s
    }

    val p = distribution(target, weights, numClasses)
    1d - sqSum(p.toArray)
  }
  private[lamp] def giniImpurityFromDistribution(
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

  private[lamp] def computeVarianceReduction(
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
