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

    def asVariable(implicit pool: Scope) =
      lamp.autograd.const(self)(pool)

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

    def normalized[S: Sc] = {
      import autograd.const
      val features = {
        val s = shape.drop(1)
        s.foldLeft(1L)(_ * _)
      }
      val weights = ATen.ones(Array(features), this.options)
      val bias = ATen.zeros(Array(features), this.options)
      val runningMean = ATen.clone(weights)
      val runningVar = ATen.clone(weights)
      val v = autograd
        .BatchNorm(
          const(self),
          const(weights),
          const(bias),
          runningMean,
          runningVar,
          true,
          1d,
          1e-5
        )
        .value
        .value

      v
    }
  }

  def inResource(f: => Tensor) = Resource.make(IO { f })(v => IO { v.release })
}
