package lamp.nn

import aten.Tensor
import lamp.autograd._
import aten.ATen
import aten.TensorOptions
import lamp.syntax

case class Residual(member: Module) extends Module {
  override def asEval: Module = copy(member.asEval)
  override def asTraining: Module = copy(member.asTraining)
  override def parameters = member.parameters
  def forward(x: Variable) = x + member.forward(x)

  override def load(parameters: Seq[Tensor]) = Residual(member.load(parameters))
}

case class Sequential(members: Module*) extends Module {
  override def asEval: Module = Sequential(members.map(_.asEval): _*)
  override def asTraining: Module = Sequential(members.map(_.asTraining): _*)
  override def parameters =
    members.zipWithIndex.flatMap {
      case (member, idx) =>
        member.parameters.map {
          case (param, ptag) => (param, Sequential.Tag(ptag, idx))
        }
    }
  def forward(x: Variable) =
    members.foldLeft(x)((x, b) => b.forward(x))

  override def load(parameters: Seq[Tensor]) = {
    val (loadedMembers, _) = members.foldLeft((List[Module](), parameters)) {
      case ((acc, params), member) =>
        val numParam = member.parameters.size
        val loaded = member.load(params.take(numParam))
        (acc.:+(loaded), params.drop(numParam))

    }
    Sequential(loadedMembers: _*)
  }
}
object Sequential {
  case class Tag[T <: PTag](t: T, idx: Int) extends PTag {
    def leaf = t
  }
}

case class Fun(fun: Variable => Variable) extends Module {
  def forward(x: Variable): Variable = fun(x)
}

trait Module {
  def asEval: Module = this
  def asTraining: Module = this
  def forward(x: Variable): Variable
  def parameters: Seq[(Variable, PTag)] = Nil
  def gradients(loss: Variable): Seq[Option[Tensor]] = {
    parameters.foreach {
      case (param, tag) =>
        param.zeroGrad()
    }
    loss.backprop()
    val g = parameters.map { case (param, _) => param.partialDerivative }
    loss.releaseAll
    g
  }
  def load(parameters: Seq[Tensor]): Module = this
  def learnableParameters =
    parameters.filter(_._1.needsGrad).map(_._1.value.numel()).sum
}

trait PTag {
  def leaf: PTag
}
trait LeafTag extends PTag {
  def leaf: PTag = this
}
case object NoTag extends LeafTag
