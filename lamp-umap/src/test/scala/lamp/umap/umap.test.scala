package lamp.umap

import org.saddle._
import org.scalatest.funsuite.AnyFunSuite
import lamp.DoublePrecision
import lamp.CPU

// import org.nspl._
// import org.nspl.awtrenderer._
// import org.nspl.saddle._

class UmapSuite extends AnyFunSuite {

  test("edge weights") {
    val data = Mat(Vec(1d, 2d, 3d), Vec(4d, 5d, 6d))
    val knn = lamp.knn.knnSearch(
      data,
      data,
      3,
      lamp.knn.SquaredEuclideanDistance,
      CPU,
      DoublePrecision,
      100
    )

    val b = Umap.edgeWeights(data, knn)
    val exp = Mat(
      Vec(0.0, 1.0, 1.0),
      Vec(0.0, 2.0, 0.0),
      Vec(1.0, 0.0, 1.0),
      Vec(1.0, 2.0, 1.0),
      Vec(2.0, 1.0, 1.0),
      Vec(2.0, 0.0, 0.0)
    ).T
    assert(b == exp)
  }

  test("mnist") {
    val data = org.saddle.csv.CsvParser
      .parseSourceWithHeader[Double](
        scala.io.Source
          .fromInputStream(
            new java.util.zip.GZIPInputStream(
              getClass.getResourceAsStream("/mnist_train.csv.gz")
            )
          ),
        maxLines = 1000
      )
      .right
      .get
      .withRowIndex(0)

    val (locs, _, loss) = Umap.umap(
      data = data.toMat,
      numDim = 2
    )

    assert(loss < 0.7)
    val _ = locs
    // println(
    //   pdfToFile(
    //     xyplot(
    //       Frame(
    //         Mat(
    //           locs.cols :+ data.rowIx.toVec: _*
    //         )
    //       ).setRowIndex(data.rowIx).colAt(0, 1, 2) -> point(
    //         size = 1d,
    //         labelText = false,
    //         color = DiscreteColors(10)
    //       )
    //     )().build
    //   )
    // )

  }
}
