package lamp
import aten.Tensor
import aten.ATen
import aten.TensorOptions
package object autograd {
  // def withPool[T](f: AllocatedVariablePool => T): T = {
  //   val pool = new AllocatedVariablePool
  //   val r = f(pool)
  //   pool.releaseAll()
  //   r
  // }

  def const[S: Sc](m: Tensor): Variable =
    Constant(m).value.needsNoGrad

  def const(
      m: Double,
      tOpt: TensorOptions = TensorOptions.dtypeDouble()
  )(implicit pool: Scope): Variable =
    Constant(ATen.scalar_tensor(m, tOpt)).value.needsNoGrad

  def param[S: Sc](m: Tensor): Variable =
    Constant(m).value
  def param[S: Sc](
      m: Double,
      tOpt: TensorOptions = TensorOptions.dtypeDouble()
  ): Variable =
    Constant(ATen.scalar_tensor(m, tOpt)).value.view(List(1, 1))

  def measure[T](tag: String)(body: => T): T = {
    val t1 = System.nanoTime()
    val r = body
    val t2 = System.nanoTime()
    println(s"$tag" + (t2 - t1) * 1e-9)
    r
  }
}
