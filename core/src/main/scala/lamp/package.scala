import aten.ATen

import cats.effect.{IO, Resource}
import lamp.autograd.TensorHelpers
import aten.Tensor
import org.saddle.Vec

package object lamp {

  implicit class syntax(self: Tensor) {

    def scalar(d: Double) = inResource {
      ATen.scalar_tensor(d, self.options())
    }

    def shape = self.sizes.toList
    def options = self.options
    def size = self.numel

    def reshape(s: Seq[Long]) = inResource(ATen.reshape(self, s.toArray))

    def select(dim: Long, index: Long) =
      inResource(ATen.select(self, dim, index))
    def select(dim: Long, index: Array[Long]) =
      inResource(TensorHelpers.fromLongVec(Vec(index), cuda = false)).flatMap(
        index => inResource(ATen.index_select(self, dim, index))
      )

    def toDoubleArray = {
      val arr = Array.ofDim[Double](size.toInt)
      val success = self.copyToDoubleArray(arr)
      assert(success)
      arr
    }
    def toMat = TensorHelpers.toMat(self)
    def toMatLong = TensorHelpers.toMatLong(self)

    def copy = inResource(ATen.clone(self))

    def *=(d: Double) = {
      scalar(d).use { t => IO { ATen.mul_out(self, self, t) } }.unsafeRunSync
    }
    def +=(other: Tensor) = {
      ATen.add_out(self, self, other, 1d)
    }

    // self += other1 * d
    def addcmul(other: Tensor, d: Double) =
      scalar(1d).use { scalarOne =>
        IO { ATen.addcmul_out(self, self, other, scalarOne, d) }
      }.unsafeRunSync

    // self += other1 * other2
    def addcmul(other1: Tensor, other2: Tensor, d: Double) =
      ATen.addcmul_out(self, self, other1, other2, d)

    def t = inResource { ATen.t(self) }
    def *(d: Double) =
      inResource { ATen.mul_1(self, d) }
    def *(other: Tensor) =
      inResource { ATen.mul_0(self, other) }
    def +(other: Tensor) =
      inResource { ATen.add_0(self, other, 1d) }
  }

  def inResource(f: => Tensor) = Resource.make(IO { f })(v => IO { v.release })

}
