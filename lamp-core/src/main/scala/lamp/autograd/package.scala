package lamp
import org.saddle.Mat
import aten.Tensor
import aten.ATen
import aten.TensorOptions
package object autograd {
  def withPool[T](f: AllocatedVariablePool => T): T = {
    val pool = new AllocatedVariablePool
    val r = f(pool)
    pool.releaseAll()
    r
  }

  def const(m: Tensor)(implicit pool: AllocatedVariablePool): Variable =
    Constant(m)(pool).value.needsNoGrad

  def const(
      m: Double,
      tOpt: TensorOptions = TensorOptions.dtypeDouble()
  )(implicit pool: AllocatedVariablePool): Variable =
    Constant(ATen.scalar_tensor(m, tOpt))(pool).value.needsNoGrad

  def param(m: Tensor)(implicit pool: AllocatedVariablePool): Variable =
    Constant(m)(pool).value

  def measure[T](tag: String)(body: => T): T = {
    val t1 = System.nanoTime()
    val r = body
    val t2 = System.nanoTime()
    println(s"$tag" + (t2 - t1) * 1e-9)
    r
  }
}
