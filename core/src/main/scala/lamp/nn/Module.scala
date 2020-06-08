package lamp.nn

import aten.Tensor
import lamp.autograd._
import aten.ATen
import aten.TensorOptions

case class Sequential(members: Module*) extends Module {
  override def asEval: Module = Sequential(members.map(_.asEval): _*)
  override def asTraining: Module = Sequential(members.map(_.asTraining): _*)
  override def parameters =
    members.flatMap(member =>
      member.parameters.zipWithIndex.map {
        case ((param, ptag), idx) => (param, Sequential.Tag(ptag, idx))
      }
    )
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
case class View(size: Int) extends Module {
  def forward(x: Variable): Variable = x.viewAs2D(size)
}
case class FlattenLast(dims: Int) extends Module {
  def forward(x: Variable): Variable = x.flattenLastDimensions(dims)
}

trait Module {
  def asEval: Module = this
  def asTraining: Module = this
  def forward(x: Variable): Variable
  def parameters: Seq[(Variable, PTag)] = Nil
  def gradients(loss: Variable): Seq[Tensor] = {
    parameters.foreach { case (param, _) => param.zeroGrad() }
    loss.backprop()
    val g = parameters.map { case (param, _) => param.partialDerivative.get }
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
