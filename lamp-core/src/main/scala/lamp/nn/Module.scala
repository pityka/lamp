package lamp.nn

import aten.Tensor
import lamp.autograd._
import aten.ATen
import aten.TensorOptions
import lamp.syntax

case class Residual[T](member: StatefulModule[T]) extends StatefulModule[T] {
  override def asEval: Residual[T] = copy(member.asEval)
  override def asTraining: Residual[T] = copy(member.asTraining)
  override def state = member.state
  def forward1(x: Variable, st: T) = {
    val (x1, st1) = member.forward1(x, st)
    val ret = x + x1
    (ret, st1)
  }

  override def load(parameters: Seq[Tensor]) = Residual(member.load(parameters))
}

case class Sequential(
    members: StatefulModule[Unit]*
) extends StatefulModule[Unit] {
  override def asEval: Sequential =
    Sequential(members.map(_.asEval): _*)
  override def asTraining: Sequential =
    Sequential(members.map(_.asTraining): _*)
  override def state =
    members.zipWithIndex.flatMap {
      case (member, idx) =>
        member.state.map {
          case (param, ptag) => (param, Sequential.Tag(ptag, idx))
        }
    }
  def forward1(x: Variable, st: Unit) =
    (members.foldLeft(x) {
      case (x, m) =>
        val (x1, _) = m.forward1(x, ())
        x1
    }, ())

  def forward(x: Variable) = forward1(x, ())._1

  override def load(tensors: Seq[Tensor]) = {
    val (loadedMembers, _) =
      members.foldLeft((List[StatefulModule[Unit]](), tensors)) {
        case ((acc, params), member) =>
          val numParam = member.state.size
          val loaded = member.load(params.take(numParam))
          (acc.:+(loaded), params.drop(numParam))

      }
    Sequential(loadedMembers: _*)
  }
}
object Sequential {
  def apply(m: StatefulModule[Unit]*): Sequential =
    Sequential.apply(m: _*)
  case class Tag[T <: PTag](t: T, idx: Int) extends PTag {
    def leaf = t
    def updateDuringOptimization: Boolean = t.updateDuringOptimization
  }
}

case class Fun(fun: Variable => Variable) extends Module {
  def forward(x: Variable): Variable = fun(x)
}

trait Module extends StatefulModule[Unit] {
  def forward(x: Variable): Variable
  override def forward1(x: Variable, state: Unit): (Variable, Unit) =
    (forward(x), ())
}

trait StatefulModule[S] {
  def asEval: StatefulModule[S] = this
  def asTraining: StatefulModule[S] = this
  def forward1(x: Variable, state: S): (Variable, S)
  def state: Seq[(Variable, PTag)] = Nil
  final def parameters =
    state.filter(v =>
      v._1.needsGrad && v._1.leaf && v._2.updateDuringOptimization
    )
  final def gradients(
      loss: Variable,
      zeroGrad: Boolean = true
  ): Seq[Option[Tensor]] = {
    if (zeroGrad) {
      parameters.foreach {
        case (param, tag) =>
          param.zeroGrad()
      }
    }
    loss.backprop()
    val g = parameters.map {
      case (param, _) => param.partialDerivative
    }
    loss.releaseAll
    g
  }
  def load(parameters: Seq[Tensor]): StatefulModule[S] = this
  final def learnableParameters =
    parameters.filter(_._1.needsGrad).map(_._1.value.numel()).sum
}

trait PTag {
  def leaf: PTag
  def updateDuringOptimization: Boolean
}
trait LeafTag extends PTag {
  def leaf: PTag = this
  def updateDuringOptimization: Boolean = true
}
trait IntermediateStateTag extends PTag {
  def leaf: PTag = this
  def updateDuringOptimization: Boolean = false
}
case object NoTag extends LeafTag
