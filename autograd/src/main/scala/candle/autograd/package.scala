package candle
import org.saddle.Mat
import aten.Tensor
package object autograd {
  def const(m: Tensor): Variable = Constant(m).value.detached
  def param(m: Tensor): Variable = Constant(m).value
}
