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

  def measure[T](tag: String)(body: => T): T = {
    val t1 = System.nanoTime()
    val r = body
    val t2 = System.nanoTime()
    println(s"$tag" + (t2 - t1) * 1e-9)
    r
  }
}
