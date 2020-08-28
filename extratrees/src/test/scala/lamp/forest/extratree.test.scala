package lamp.extratrees

import org.saddle._
import org.scalatest.funsuite.AnyFunSuite

class ExtraTreesSuite extends AnyFunSuite {
  test("variance reduction") {
    val t = Vec(0d, 0.01, 100d, 100.1)
    val r = computeVarianceReduction(
      target = t,
      samplesInSplit = Vec(0, 1),
      samplesOutSplit = Vec(2, 3),
      varianceNoSplit = t.sampleVariance * (t.length - 1d) / t.length
    )
    assert(r == 0.999999495454448)
  }
  test("gini impurity") {
    val t = Vec(1, 1, 0, 0)
    val gt = giniImpurity(t, 2)
    assert(
      giniScore(
        target = t,
        samplesInSplit = Vec(true, true, false, false),
        giniImpurityNoSplit = gt,
        2
      ) == 0.5
    )
  }
  test("splitRegression 1") {
    val r = splitRegression(
      data = Mat(Vec(0d, 2d, 3d, 4d, 5d), Vec(100d, 99d, 98d, 97d, 96d)),
      subset = Vec(0, 1, 2, 3, 4),
      attributes = Vec(0, 1),
      target = Vec(0d, 0.1d, 100d, 100.1, 100.2),
      rng = org.saddle.spire.random.rng.Cmwc5.fromTime(0L)
    )
    assert(r == ((1, 98.739216819089)))
  }
  test("splitClassification 1") {
    val r = splitClassification(
      data = Mat(Vec(0d, 2d, 3d, 4d, 5d), Vec(100d, 99d, 98d, 97d, 96d)),
      subset = Vec(0, 1, 2, 3, 4),
      attributes = Vec(0, 1),
      targetAtSubset = Vec(1, 1, 0, 0, 0),
      rng = org.saddle.spire.random.rng.Cmwc5.fromTime(0L),
      2
    )
    assert(r == ((1, 98.739216819089)))
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
      .right
      .get
    val target =
      Mat(data.firstCol("label").toVec.map(_.toLong))
    val features = data.filterIx(_ != "label").toMat
    val t1 = System.nanoTime
    val trees = buildForestClassification(
      data = features,
      target = target.col(0).map(_.toInt),
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
  ignore("mnist full, slow ") {
    val data = org.saddle.csv.CsvParser
      .parseSourceWithHeader[Double](
        scala.io.Source
          .fromInputStream(
            new java.util.zip.GZIPInputStream(
              getClass.getResourceAsStream("/mnist_train.csv.gz")
            )
          )
      )
      .right
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
      .right
      .get
    val target =
      Mat(data.firstCol("label").toVec.map(_.toLong))
    val features = data.filterIx(_ != "label").toMat
    val testfeatures = datatest.filterIx(_ != "label").toMat
    val t1 = System.nanoTime
    val trees = buildForestClassification(
      data = features,
      target = target.col(0).map(_.toInt),
      numClasses = 10,
      nMin = 2,
      k = 32,
      m = 10,
      parallelism = 4
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
    assert(accuracy > 0.93)
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
      .right
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
      parallelism = 1
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
