package lamp
import scala.language.implicitConversions
import lamp.autograd.Variable
import aten.Tensor
import aten.ATen
import cats.effect.IO

package object nn {
  implicit def funToModule(fun: Variable => Variable) = Fun(fun)

  def gradientClippingInPlace(
      gradients: Seq[Option[Tensor]],
      theta: Double
  ): Unit = {
    val norm = math.sqrt(gradients.map {
      case Some(g) =>
        val tmp = ATen.pow_0(g, 2d)
        val d = ATen.sum_0(tmp).toMat.raw(0)
        tmp.release
        d
      case None => 0d
    }.sum)
    if (norm > theta) {
      gradients.foreach {
        case None =>
        case Some(g) =>
          g.scalar(theta / norm)
            .use { scalar => IO { ATen.mul_out(g, g, scalar) } }
            .unsafeRunSync()
      }
    }
  }
}
