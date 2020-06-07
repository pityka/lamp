package lamp
import scala.language.implicitConversions
import lamp.autograd.Variable
import aten.Tensor
import aten.ATen
import cats.effect.IO

package object nn {
  implicit def funToModule(fun: Variable => Variable) = Fun(fun)

  def gradientClippingInPlace(gradients: Seq[Tensor], theta: Double): Unit = {
    val norm = math.sqrt(gradients.map { g =>
      val tmp = ATen.pow_0(g, 2d)
      val d = ATen.sum_0(tmp).toMat.raw(0)
      tmp.release
      d
    }.sum)
    if (norm > theta) {
      gradients.foreach { g =>
        g.scalar(theta / norm)
          .use { scalar => IO { ATen.mul_out(g, g, scalar) } }
          .unsafeRunSync()
      }
    }
  }
}
