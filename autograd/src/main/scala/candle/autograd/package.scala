package candle
import org.saddle.Mat
import aten.Tensor
import aten.ATen
import aten.TensorOptions
package object autograd {
  def const(m: Tensor): Variable = Constant(m).value.detached
  def const(m: Double): Variable =
    Constant(ATen.scalar_tensor(m, TensorOptions.dtypeDouble())).value.detached
  def param(m: Tensor): Variable = Constant(m).value
}
