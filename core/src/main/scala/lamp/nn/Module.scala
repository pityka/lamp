package lamp.nn

import aten.Tensor
import lamp.autograd._
import aten.ATen
import aten.TensorOptions

object TrainLoop {
  def simple(
      module: Module,
      data1: Tensor,
      target1: Tensor,
      optimizerFactory: Seq[(Variable, PTag)] => Optimizer,
      epochs: Int
  ) = {
    val optim = optimizerFactory(module.parameters)
    val data: Variable = const(data1)
    val target: Variable = const(target1)

    var i = 0
    while (i < epochs) {
      val output = module.forward(data)
      val loss: Variable = (output - target).pow(2d).sum
      val gradients = module.gradients(loss)
      optim.step(gradients)
      i += 1
    }
  }
}

case class Sequential(members: Module*) extends Module {
  def parameters =
    members.flatMap(member =>
      member.parameters.zipWithIndex.map {
        case ((param, ptag), idx) => (param, Sequential.Tag(ptag, idx))
      }
    )
  def forward(x: Variable) =
    members.foldLeft(x)((x, b) => b.forward(x))
}
object Sequential {
  case class Tag[T <: PTag](t: T, idx: Int) extends PTag {
    def leaf = t
  }
}

case class Fun(fun: Variable => Variable) extends Module {
  def parameters = Nil
  def forward(x: Variable): Variable = fun(x)
}

trait Module {
  def forward(x: Variable): Variable
  def parameters: Seq[(Variable, PTag)]
  def gradients(loss: Variable): Seq[Tensor] = {
    parameters.foreach { case (param, _) => param.zeroGrad() }
    loss.backprop()
    val g = parameters.map { case (param, _) => param.partialDerivative.get }
    loss.release
    g
  }
}

trait PTag {
  def leaf: PTag
}
trait LeafTag extends PTag {
  def leaf: PTag = this
}
case object NoTag extends LeafTag
