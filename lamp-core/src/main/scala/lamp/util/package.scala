package lamp

import aten.{ATen, Tensor}
import org.saddle._
import cats.effect.Resource
import cats.effect.IO

package object util {
  implicit class syntax(self: Tensor) {

    def scalar(d: Double) = inResource {
      ATen.scalar_tensor(d, self.options())
    }

    def copy = ATen.clone(self)

    def shape = self.sizes.toList
    def options = self.options
    def size = self.numel

    def toDoubleArray = {
      val arr = Array.ofDim[Double](size.toInt)
      val success = self.copyToDoubleArray(arr)
      assert(success)
      arr
    }
    def toMat: Mat[Double] = {
      val opt = self.options
      if (opt.isDouble())
        TensorHelpers.toMat(self)
      else if (opt.isFloat()) TensorHelpers.toFloatMat(self).map(_.toDouble)
      else
        throw new RuntimeException(
          "Expected Double or Float tensor. got: " + opt.scalarTypeByte()
        )
    }
    def toLongMat = TensorHelpers.toLongMat(self)
    def toLongVec = TensorHelpers.toLongMat(self).toVec

  }

  def inResource(f: => Tensor) = Resource.make(IO { f })(v => IO { v.release })
}
