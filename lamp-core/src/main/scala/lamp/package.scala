import aten.ATen

import cats.effect.{IO, Resource}
import lamp.autograd.{TensorHelpers, AllocatedVariablePool}
import aten.Tensor
import org.saddle.Mat

package object lamp {

  implicit class syntax(self: Tensor) {

    def scalar(d: Double) = inResource {
      ATen.scalar_tensor(d, self.options())
    }

    def shape = self.sizes.toList
    def options = self.options
    def size = self.numel

    def asVariable(implicit pool: AllocatedVariablePool) =
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

    def normalized = inResource {
      import autograd.const
      val features = {
        val s = shape.drop(1)
        s.foldLeft(1L)(_ * _)
      }
      val weights = ATen.ones(Array(features), this.options)
      val bias = ATen.zeros(Array(features), this.options)
      val runningMean = ATen.clone(weights)
      val runningVar = ATen.clone(weights)
      val pool = new AllocatedVariablePool
      val v = autograd
        .BatchNorm(
          const(self)(pool),
          const(weights)(pool),
          const(bias)(pool),
          runningMean,
          runningVar,
          true,
          1d,
          1e-5
        )
        .value
        .value
      weights.release
      bias.release
      runningVar.release
      runningMean.release
      v
    }
  }

  def inResource(f: => Tensor) = Resource.make(IO { f })(v => IO { v.release })

}
