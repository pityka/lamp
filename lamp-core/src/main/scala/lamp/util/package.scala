package lamp

import aten.{ATen, Tensor}
// import org.saddle._
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

    def toDoubleArray = TensorHelpers.toDoubleArray(self)
    def toFloatArray = TensorHelpers.toFloatArray(self)
    def toLongArray = TensorHelpers.toLongArray(self)
    

  }

  def inResource(f: => Tensor) = Resource.make(IO { f })(v => IO { v.release })
}
