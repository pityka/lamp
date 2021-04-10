package lamp.extratrees

import org.saddle._
import org.scalatest.funsuite.AnyFunSuite

class ExtraTreesSuite extends AnyFunSuite {
  test("variance reduction") {
    val t = Vec(0d, 0.01, 100d, 100.1)
    val r = computeVarianceReduction(
      target = t,
      samplesInSplit = Vec(true, true, false, false),
      varianceNoSplit = t.sampleVariance * (t.length - 1d) / t.length
    )
    assert(r == 0.999999495454448)
  }
  test("gini impurity") {
    val t = Vec(1, 1, 0, 0)
    val gt = giniImpurity(t, None, 2)
    assert(
      giniScore(
        target = t,
        sampleWeights = None,
        samplesInSplit = Vec(true, true, false, false),
        giniImpurityNoSplit = gt,
        2,
        Array.ofDim[Double](2),
        Array.ofDim[Double](2)
      ) == 0.5
    )
  }
  test("gini impurity weighted") {
    val t = Vec(1, 1, 0, 0)
    val gt = giniImpurity(t, Some(vec.ones(4)), 2)
    assert(
      giniScore(
        target = t,
        sampleWeights = None,
        samplesInSplit = Vec(true, true, false, false),
        giniImpurityNoSplit = gt,
        2,
        Array.ofDim[Double](2),
        Array.ofDim[Double](2)
      ) == 0.5
    )
  }
  test("gini impurity 0-weighted") {
    val t = Vec(1, 1, 0, 0)
    val gt = giniImpurity(t, Some(vec.zeros(4)), 2)
    assert(
      giniScore(
        target = t,
        sampleWeights = Some(vec.zeros(4)),
        samplesInSplit = Vec(true, true, false, false),
        giniImpurityNoSplit = gt,
        2,
        Array.ofDim[Double](2),
        Array.ofDim[Double](2)
      ).isNaN
    )
  }
  test("gini impurity 0-weighted 2") {
    val t = Vec(1, 1, 0, 0)
    val gt = giniImpurity(t, Some(vec.ones(4)), 2)
    assert(
      giniScore(
        target = t,
        sampleWeights = Some(Vec(1d, 1d, 0d, 0d)),
        samplesInSplit = Vec(true, true, false, false),
        giniImpurityNoSplit = gt,
        2,
        Array.ofDim[Double](2),
        Array.ofDim[Double](2)
      ).isNaN
    )
  }
  test("splitRegression 1") {
    val attr = Array(0, 1)
    val r = splitRegression(
      data = Mat(Vec(0d, 2d, 3d, 4d, 5d), Vec(100d, 99d, 98d, 97d, 96d)),
      subset = Vec(0, 1, 2, 3, 4),
      attributes = attr,
      numConstant = 0,
      k = 2,
      targetAtSubset = Vec(0d, 0.1d, 100d, 100.1, 100.2),
      rng = org.saddle.spire.random.rng.Cmwc5.fromTime(0L)
    )
    assert(r == ((0, 3.424021023861243, 0)))
  }
  test("splitClassification 1") {
    val attr = Array(0, 1)
    val r = splitClassification(
      data = Mat(Vec(0d, 2d, 3d, 4d, 5d), Vec(100d, 99d, 98d, 97d, 96d)),
      subset = Vec(0, 1, 2, 3, 4),
      attributes = attr,
      numConstant = 0,
      k = 2,
      numClasses = 2,
      targetAtSubset = Vec(1, 1, 0, 0, 0),
      weightsAtSubset = None,
      rng = org.saddle.spire.random.rng.Cmwc5.fromTime(0L)
    )
    assert(attr.toVector == Vector(1, 0))
    assert(r == ((0, 3.424021023861243, 0)))
  }
  test("splitClassification 1 weighted") {
    val attr = Array(0, 1)
    val r = splitClassification(
      data = Mat(Vec(0d, 2d, 3d, 4d, 5d), Vec(100d, 99d, 98d, 97d, 96d)),
      subset = Vec(0, 1, 2, 3, 4),
      attributes = attr,
      numConstant = 0,
      k = 2,
      numClasses = 2,
      targetAtSubset = Vec(1, 1, 0, 0, 0),
      weightsAtSubset = Some(vec.ones(5)),
      rng = org.saddle.spire.random.rng.Cmwc5.fromTime(0L)
    )
    assert(attr.toVector == Vector(1, 0))
    assert(r == ((0, 3.424021023861243, 0)))
  }
  test("splitClassification 1 0-weighted") {
    val attr = Array(0, 1)
    val r = splitClassification(
      data = Mat(Vec(0d, 2d, 3d, 4d, 5d), Vec(100d, 99d, 98d, 97d, 96d)),
      subset = Vec(0, 1, 2, 3, 4),
      attributes = attr,
      numConstant = 0,
      k = 2,
      numClasses = 2,
      targetAtSubset = Vec(1, 1, 0, 0, 0),
      weightsAtSubset = Some(Vec(1d, 1d, 0d, 0d, 0d)),
      rng = org.saddle.spire.random.rng.Cmwc5.fromTime(0L)
    )
    assert(attr.toVector == Vector(0, 1))
    assert(r._1 == -1)
  }
  test("splitClassification 2") {
    val attr = Array(0, 1)
    val r = splitClassification(
      data = Mat(Vec(0d, 0d, 3d, 3d, 3d), Vec(100d, 99d, 98d, 97d, 96d)),
      subset = Vec(2, 3, 4),
      attributes = attr,
      numConstant = 0,
      k = 2,
      numClasses = 2,
      targetAtSubset = Vec(1, 1, 0),
      weightsAtSubset = None,
      rng = org.saddle.spire.random.rng.Cmwc5.fromTime(0L)
    )
    assert(r == ((1, 97.54668482609304, 1)))
    assert(attr.toVector == Vector(0, 1))
  }
  test("splitClassification 3") {
    val attr = Array(2, 1, 0)
    val r = splitClassification(
      data = Mat(
        Vec(0d, 0d, 3d, 3d, 3d),
        Vec(0d, 0d, 3d, 3d, 3d),
        Vec(100d, 99d, 98d, 97d, 96d)
      ),
      subset = Vec(2, 3, 4),
      attributes = attr,
      numConstant = 0,
      k = 2,
      numClasses = 2,
      targetAtSubset = Vec(1, 1, 0),
      weightsAtSubset = None,
      rng = org.saddle.spire.random.rng.Cmwc5.fromTime(0L)
    )
    assert(r == ((2, 97.54668482609304, 2)))
    assert(attr.toVector == Vector(0, 1, 2))
  }
  test("splitClassification 4") {
    val attr = Array(2, 0, 1)
    val r = splitClassification(
      data = Mat(
        Vec(0d, 0d, 3d, 3d, 3d),
        Vec(0d, 0d, 3d, 3d, 3d),
        Vec(100d, 99d, 98d, 97d, 96d)
      ),
      subset = Vec(2, 3, 4),
      attributes = attr,
      numConstant = 0,
      k = 1,
      numClasses = 2,
      targetAtSubset = Vec(1, 1, 0),
      weightsAtSubset = None,
      rng = org.saddle.spire.random.rng.Cmwc5.fromTime(0L)
    )
    assert(r == ((2, 97.54668482609304, 1)))
    assert(attr.toVector == Vector(1, 0, 2))
  }
  test("splitClassification 5") {
    val attr = Array(0, 2, 1)
    val r = splitClassification(
      data = Mat(
        Vec(0d, 0d, 3d, 3d, 3d),
        Vec(0d, 0d, 3d, 3d, 3d),
        Vec(100d, 99d, 98d, 97d, 96d)
      ),
      subset = Vec(2, 3, 4),
      attributes = attr,
      numConstant = 1,
      k = 1,
      numClasses = 2,
      targetAtSubset = Vec(1, 1, 0),
      weightsAtSubset = None,
      rng = org.saddle.spire.random.rng.Cmwc5.fromTime(1L)
    )
    assert(r == ((2, 97.84900936098786, 1)))
    assert(attr.toVector == Vector(0, 1, 2))
  }
  test("splitClassification 6") {
    val attr = Array(0, 2, 1)
    val r = splitClassification(
      data = Mat(
        Vec(0d, 0d, 3d, 3d, 3d),
        Vec(0d, 0d, 3d, 3d, 3d),
        Vec(100d, 99d, 98d, 97d, 96d)
      ),
      subset = Vec(2, 3, 4),
      attributes = attr,
      numConstant = 1,
      k = 1,
      numClasses = 2,
      targetAtSubset = Vec(1, 1, 0),
      weightsAtSubset = None,
      rng = org.saddle.spire.random.rng.Cmwc5.fromTime(123L)
    )
    assert(r == ((2, 96.07259095141863, 2)))
    assert(attr.toVector == Vector(0, 1, 2))
  }
  test("splitClassification 7") {
    val attr = Array(1, 2, 0)
    val r = splitClassification(
      data = Mat(
        Vec(0d, 0d, 3d, 3d, 3d),
        Vec(0d, 0d, 3d, 3d, 3d),
        Vec(100d, 99d, 98d, 97d, 96d)
      ),
      subset = Vec(2, 3, 4),
      attributes = attr,
      numConstant = 1,
      k = 1,
      numClasses = 2,
      targetAtSubset = Vec(1, 1, 0),
      weightsAtSubset = None,
      rng = org.saddle.spire.random.rng.Cmwc5.fromTime(123L)
    )
    assert(r == ((2, 96.07259095141863, 2)))
    assert(attr.toVector == Vector(1, 0, 2))
  }

  test("mnist") {
    val data = org.saddle.csv.CsvParser
      .parseSourceWithHeader[Double](
        scala.io.Source
          .fromInputStream(
            new java.util.zip.GZIPInputStream(
              getClass.getResourceAsStream("/mnist_test.csv.gz")
            )
          )
      )
      .toOption
      .get
    val target =
      Mat(data.firstCol("label").toVec.map(_.toLong))
    val features = data.filterIx(_ != "label").toMat
    val t1 = System.nanoTime
    val trees = buildForestClassification(
      data = features,
      target = target.col(0).map(_.toInt),
      sampleWeights = None,
      numClasses = 10,
      nMin = 2,
      k = 32,
      m = 1,
      parallelism = 1
    )
    println((System.nanoTime() - t1) * 1e-9)
    val output = predictClassification(trees, features)
    val prediction = {
      output.rows.map(_.argmax).toVec
    }
    val correct =
      prediction.zipMap(data.firstCol("label").toVec.map(_.toInt))((a, b) =>
        if (a == b) 1d else 0d
      )
    val accuracy = correct.mean2
    assert(accuracy == 1.0)
  }
  test("mnist weighted") {
    val data = org.saddle.csv.CsvParser
      .parseSourceWithHeader[Double](
        scala.io.Source
          .fromInputStream(
            new java.util.zip.GZIPInputStream(
              getClass.getResourceAsStream("/mnist_test.csv.gz")
            )
          )
      )
      .toOption
      .get
    val target =
      Mat(data.firstCol("label").toVec.map(_.toLong))
    val features = data.filterIx(_ != "label").toMat
    val t1 = System.nanoTime
    val trees = buildForestClassification(
      data = features,
      target = target.col(0).map(_.toInt),
      sampleWeights = Some(
        vec.ones(features.numRows / 2) concat vec.zeros(features.numRows / 2)
      ),
      numClasses = 10,
      nMin = 2,
      k = 32,
      m = 1,
      parallelism = 8
    )
    println((System.nanoTime() - t1) * 1e-9)
    val output = predictClassification(trees, features)
    val prediction = {
      output.rows.map(_.argmax).toVec
    }
    val correct =
      prediction.zipMap(data.firstCol("label").toVec.map(_.toInt))((a, b) =>
        if (a == b) 1d else 0d
      )
    val accuracy = correct.mean2
    assert(accuracy <= 0.9)
  }
  test("mnist full, slow ") {
    val data = org.saddle.csv.CsvParser
      .parseSourceWithHeader[Double](
        scala.io.Source
          .fromInputStream(
            new java.util.zip.GZIPInputStream(
              getClass.getResourceAsStream("/mnist_train.csv.gz")
            )
          )
      )
      .toOption
      .get
    val datatest = org.saddle.csv.CsvParser
      .parseSourceWithHeader[Double](
        scala.io.Source
          .fromInputStream(
            new java.util.zip.GZIPInputStream(
              getClass.getResourceAsStream("/mnist_test.csv.gz")
            )
          )
      )
      .toOption
      .get
    val target =
      Mat(data.firstCol("label").toVec.map(_.toLong))
    val features = data.filterIx(_ != "label").toMat
    val testfeatures = datatest.filterIx(_ != "label").toMat
    val t1 = System.nanoTime
    val trees = buildForestClassification(
      data = features,
      target = target.col(0).map(_.toInt),
      sampleWeights = None,
      numClasses = 10,
      nMin = 2,
      k = 32,
      m = 10,
      parallelism = 8
    )
    println((System.nanoTime() - t1) * 1e-9)
    val output = predictClassification(trees, testfeatures)
    val prediction = {
      output.rows.map(_.argmax).toVec
    }
    val correct =
      prediction.zipMap(datatest.firstCol("label").toVec.map(_.toInt))((a, b) =>
        if (a == b) 1d else 0d
      )
    val accuracy = correct.mean2
    assert(accuracy > 0.95)
  }
  test("mnist regression") {
    val data = org.saddle.csv.CsvParser
      .parseSourceWithHeader[Double](
        scala.io.Source
          .fromInputStream(
            new java.util.zip.GZIPInputStream(
              getClass.getResourceAsStream("/mnist_test.csv.gz")
            )
          )
      )
      .toOption
      .get
    val target =
      Mat(data.firstCol("label").toVec.map(_.toLong)).toVec.map(_.toDouble)
    val features = data.filterIx(_ != "label").toMat
    val t1 = System.nanoTime()
    val trees = buildForestRegression(
      data = features,
      target = target,
      nMin = 2,
      k = 32,
      m = 1,
      parallelism = 8
    )
    println((System.nanoTime - t1) * 1e-9)
    val output = predictRegression(trees, features)
    val prediction = {
      output.map(_.toInt)
    }
    val correct =
      prediction.zipMap(data.firstCol("label").toVec.map(_.toInt))((a, b) =>
        if (a == b) 1d else 0d
      )
    val accuracy = correct.mean2
    assert(accuracy == 1.0)
  }

}
