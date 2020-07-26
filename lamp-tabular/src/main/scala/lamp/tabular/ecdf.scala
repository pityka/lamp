package lamp.tabular

import org.saddle._
import org.saddle.ops.BinOps._
import aten.Tensor
import lamp.autograd.TensorHelpers
import cats.effect.Resource
import cats.effect.IO

case class ECDF(x: Vec[Double], y: Vec[Double]) {
  def apply(n: Double) = {
    val idx = x.findOne(_ > n)
    if (idx == -1) 1d
    else if (idx == 0) 0d
    else y.raw(idx - 1)
  }
  def apply(v: Vec[Double]): Vec[Double] = v.map(this.apply)

  def inverse(v: Vec[Double]): Vec[Double] = v.map(this.inverse)

  def inverse(n: Double) = {
    val idx = y.findOne(_ >= n)
    if (idx == -1) x.max2
    else if (idx == 0) x.min2
    else if (y.raw(idx) == n) x.raw(idx)
    else {
      val idx2 = idx
      val idx1 = idx - 1
      val v1 = x.raw(idx1)
      val v2 = x.raw(idx2)
      val w1 = y.raw(idx1)
      val w2 = y.raw(idx2)
      ((n - w1) / (w2 - w1)) * (v2 - v1) + v1
    }
  }
  def apply(t: Tensor): Resource[IO, Tensor] =
    Resource.make {
      IO {
        import lamp.syntax
        val v = t.toMat.col(0)
        val transformed = v.map(x => this(x))
        val t2 = TensorHelpers.fromVec(
          transformed,
          TensorHelpers.device(t),
          TensorHelpers.precision(t).get
        )
        t2
      }
    }(v => IO { v.release })
}

object ECDF {
  def apply(t: Tensor): ECDF = {
    import lamp.syntax
    val v = t.toMat.col(0)
    ECDF(v)

  }
  def apply(xs: Vec[Double]): ECDF = {
    val sorted = xs.sorted
    val ranks = sorted.rank(tie = RankTie.Max) / xs.length
    ECDF(sorted, ranks)
  }
}
