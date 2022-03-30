package lamp.knn

import org.saddle._
import org.saddle.linalg._
import org.saddle.ops.BinOps._
import org.scalatest.funsuite.AnyFunSuite
import lamp.DoublePrecision
import lamp.CPU
import lamp.saddle._
import lamp.nn.CudaTest
import lamp.CudaDevice
import lamp.Scope

class KnnSuite extends AnyFunSuite {
  def test1(id: String)(fun: Boolean => Unit) = {
    test(id) { fun(false) }
    test(id + "/CUDA", CudaTest) { fun(true) }
  }

  test1("euclidean") { cuda =>
    Scope.root { implicit scope =>
      val m3x2 = Mat(Vec(0d, 2d, 3d), Vec(1d, 0d, 1d))
      val t = lamp.saddle.fromMat(m3x2, cuda)
      val d = squaredEuclideanDistance(t, t).toMat
      assert(d.toVec == m3x2.rows.flatMap { rowi =>
        m3x2.rows.map { rowj =>
          val v = rowi - rowj
          v vv v
        }
      }.toVec)
    }
  }
  test1("jaccard") { cuda =>
    Scope.root { implicit scope =>
      val m2x2 = Mat(Vec(0d, 1d), Vec(1d, 1d))
      val t = lamp.saddle.fromMat(m2x2, cuda)
      val d = jaccardDistance(t, t).toMat
      assert(d.toVec == m2x2.rows.flatMap { rowi =>
        m2x2.rows.map { rowj =>
          val v =
            rowi.zipMap(rowj)((a, b) => if (a == 1d && b == 1d) 1d else 0d).sum
          val v2 =
            rowi.zipMap(rowj)((a, b) => if (a == 1d || b == 1d) 1d else 0d).sum
          1d - v / v2
        }
      }.toVec)
    }
  }
  test1("knnSearch") { cuda =>
    val data =
      Mat(Vec(0d, 0d), Vec(50d, 50d), Vec(1000d, 1000d), Vec(1d, 1d)).T
    val query = Mat(Vec(1d, 1d), Vec(495d, 495d)).T
    val indices = knnSearch(
      data,
      query,
      2,
      SquaredEuclideanDistance,
      if (cuda) CudaDevice(0) else CPU,
      DoublePrecision,
      1
    )
    assert(indices.rows.map(_.toSeq.toSet) == Seq(Set(0, 3), Set(1, 3)))

  }
  test("classification") {
    val values = Vec(0, 1, 2)
    val indices = Mat(Vec(0, 1), Vec(1, 2)).T
    assert(
      classification(values, indices, 3, log = false) == Mat(
        Vec(0.5, 0.5, 0d),
        Vec(0d, 0.5, 0.5)
      ).T
    )
  }
}
