package lamp.umap

import org.saddle._
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

    val knnDistances = knn.mapRows { case (row, rowIdx) =>
      val row1 = data.row(rowIdx)
      row.map { idx2 =>
        val row2 = data.row(idx2)
                val d = {
          var i = 0
          val l = row1.length
          var s = 0d
          val r1a = row1.toArray
          val r2a = row2.toArray
          while (i < l) {
            val d = r1a(i) - r2a(i)
            s += d * d
            i += 1
          }
          s
        }
        math.sqrt(d)
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

    assert(loss < 0.7)
    val _ = locs

  }
}
