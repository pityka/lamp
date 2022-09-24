package lamp.kmeans

import lamp.saddle._
import org.saddle._
import org.scalatest.funsuite.AnyFunSuite
import lamp.nn.CudaTest
import lamp.Scope
import lamp.STen
import lamp.CudaDevice

class KmeansSuite extends AnyFunSuite {
  def test1(id: String)(fun: Boolean => Unit) = {
    test(id) { fun(false) }
    test(id + "/CUDA", CudaTest) { fun(true) }
  }

  test1("simple") { cuda =>
    Scope.unsafe { implicit scope =>
      val c1 = STen.randn(List(100, 2))
      val c2 =
        STen.randn(List(100, 2)) + lamp.saddle.fromMat(Mat(Vec(100d, 100d)).T)
      val c3 =
        STen.randn(List(100, 2)) + lamp.saddle.fromMat(Mat(Vec(200d, 200d)).T)
      val c4 =
        STen.randn(List(100, 2)) + lamp.saddle.fromMat(Mat(Vec(300d, 300d)).T)
      val d = STen.cat(List(c1, c2, c3, c4), dim = 0)
      val d2 = if (cuda) CudaDevice(0).to(d) else d
      val (centers, members, distances) = minibatchKMeans(
        instances = d2,
        clusters = 4,
        iterations = 5,
        minibatchSize = 10,
        learningRate = 0.1
      )
      val centersM = centers.toMat.rows

      def assertCenter(c: Vec[Double],min:Int,max:Int) = {
        import org.saddle.ops.BinOps._

        val p = centersM.zipWithIndex.find { case (row,_) =>
          val e = row - c
          val d = math.sqrt((e dot e))
          d < 10d
        }
        assert(p.isDefined)
        assert(members.toLongVec.slice(min,max).toSeq.forall(_ == p.get._2))

      }

      assertCenter(Vec(0d, 0d),0,100)
      assertCenter(Vec(100d, 100d),100,200)
      assertCenter(Vec(200d, 200d),200,300)
      assertCenter(Vec(300d, 300d),300,400)

      assert(distances.toVec.toSeq.forall(_ < 100d))

    }
  }

}
