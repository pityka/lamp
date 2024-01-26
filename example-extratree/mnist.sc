//> using scala 2.13
//> using lib io.github.pityka::extratrees::0.0.111
//> using lib io.github.pityka::fileutils::1.2.5

import org.saddle._
import lamp.extratrees._
val mnist_test = "../lamp-core/src/test/resources/mnist_test.csv.gz"
val mnist_train = "../lamp-core/src/test/resources/mnist_train.csv.gz"

val trees = {
  val data = fileutils
    .openSource(mnist_train)(src =>
      org.saddle.csv.CsvParser
        .parseSourceWithHeader[Double](src)
    )
    .toOption
    .get

  val target =
    Mat(data.firstCol("label").toVec.map(_.toLong))
  val features = data.filterIx(_ != "label").toMat
  buildForestClassification(
    data = features,
    target = target.col(0).map(_.toInt),
    sampleWeights = None,
    numClasses = 10,
    nMin = 2, // min leaf size
    k = 1, // choose the best among k random splits. k=1 is totally random
    m = 100, // num trees
    parallelism = 4
  )
}

val datatest = fileutils
  .openSource(mnist_test)(src =>
    org.saddle.csv.CsvParser
      .parseSourceWithHeader[Double](src)
  )
  .toOption
  .get

val featurestest = datatest.filterIx(_ != "label").toMat
val output = predictClassification(trees, featurestest)
val prediction = {
  output.rows.map(_.argmax).toVec
}
val correct =
  prediction.zipMap(datatest.firstCol("label").toVec.map(_.toInt))((a, b) =>
    if (a == b) 1d else 0d
  )
val accuracy = correct.mean2
println("test accuracy:")
println(accuracy)
