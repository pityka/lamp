package lamp
import org.saddle.Mat
import aten.Tensor
import aten.ATen
import aten.TensorOptions
package object autograd {
  def const(m: Tensor): Variable = Constant(m).value.needsNoGrad
  def const(
      m: Double,
      tOpt: TensorOptions = TensorOptions.dtypeDouble()
  ): Variable =
    Constant(ATen.scalar_tensor(m, tOpt)).value.needsNoGrad
  def param(m: Tensor): Variable = Constant(m).value
}
