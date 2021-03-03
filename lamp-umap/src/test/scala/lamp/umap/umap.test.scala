package lamp.umap

import org.saddle._
import org.saddle.linalg._
import org.saddle.macros.BinOps._
import org.scalatest.funsuite.AnyFunSuite
import lamp.DoublePrecision
import lamp.CPU

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

    val knnDistances = knn.mapRows {
      case (row, rowIdx) =>
        val row1 = data.row(rowIdx)
        row.map { idx2 =>
          val row2 = data.row(idx2)
          val d = row1 - row2
          math.sqrt(d vv d)
        }
    }

    val b = Umap.edgeWeights(knnDistances, knn)
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
      .toOption
      .get
      .withRowIndex(0)

    val (locs, _, loss) = Umap.umap(
      data = data.toMat,
      numDim = 2,
      positiveSamples = Some(5000),
      negativeSampleSize = 5,
      // logger = Some(scribe.Logger("sfa")),
      iterations = 1000
    )

    // import org.nspl._
    // import org.nspl.awtrenderer._
    // import org.nspl.saddle._
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

    assert(loss < 0.7)
    val _ = locs

  }
}
