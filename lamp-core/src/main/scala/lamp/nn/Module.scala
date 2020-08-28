package lamp.nn

import aten.Tensor
import lamp.autograd._

case class Sequential[A, M <: GenericModule[A, A]](
    members: M with GenericModule[A, A]*
) extends GenericModule[A, A] {
  override def state =
    members.zipWithIndex.flatMap {
      case (member, idx) =>
        member.state.map {
          case (param, ptag) => (param, Sequential.Tag(ptag, idx))
        }
    }
  def forward(x: A) =
    members.foldLeft(x) {
      case (x, m) =>
        m.forward(x)
    }

}
object Sequential {

  implicit def trainingMode[A, M <: GenericModule[A, A]: TrainingMode] =
    TrainingMode.make[Sequential[A, M]](
      module => Sequential(module.members.map(_.asEval): _*),
      module => Sequential(module.members.map(_.asTraining): _*)
    )

  implicit def load[A, M <: GenericModule[A, A]: Load] =
    Load.make[Sequential[A, M]] { module => tensors =>
      val (loadedMembers, _) =
        module.members.foldLeft((List[M](), tensors)) {
          case ((acc, params), member) =>
            val numParam = member.state.size
            val loaded = member.load(params.take(numParam))
            (acc.:+(loaded), params.drop(numParam))

        }
      Sequential(loadedMembers: _*)
    }

  case class Tag[T <: PTag](t: T, idx: Int) extends PTag {
    def leaf = t
    def updateDuringOptimization: Boolean = t.updateDuringOptimization
  }
}

case class Fun(fun: Variable => Variable) extends Module {
  def state = Nil
  def forward(x: Variable): Variable = fun(x)
}
object Fun {
  implicit val trainingMode = TrainingMode.identity[Fun]
  implicit val load = Load.identity[Fun]
}

case class GenericFun[A, B](fun: A => B) extends GenericModule[A, B] {
  def state = Nil
  def forward(x: A): B = fun(x)
}
object GenericFun {
  implicit def trainingMode[A, B] = TrainingMode.identity[GenericFun[A, B]]
  implicit def load[A, B] = Load.identity[GenericFun[A, B]]
}

case class LiftedModule[M <: Module](mod: M with Module)
    extends StatefulModule[Variable, Variable, Unit] {
  def state = mod.state
  def forward(x: (Variable, Unit)) = (mod.forward(x._1), ())
}
object LiftedModule {
  implicit def trainingMode[
      M <: Module: TrainingMode
  ] =
    TrainingMode.make[LiftedModule[M]](
      m => m.copy(mod = m.mod.asEval),
      m => m.copy(mod = m.mod.asTraining)
    )
  implicit def load[
      M <: Module: Load
  ] =
    Load.make[LiftedModule[M]](m =>
      tensors => m.copy(mod = m.mod.load(tensors))
    )
  implicit def initState[M <: Module] =
    InitState.make[LiftedModule[M], Unit](_ => ())
}

case class UnliftedModule[A, B, C, D, M <: StatefulModule2[A, B, C, D]](
    statefulModule: M with StatefulModule2[A, B, C, D]
)(implicit init: InitState[M, C])
    extends GenericModule[A, B] {
  def state = statefulModule.state
  def forward(x: A) = statefulModule.forward((x, statefulModule.initState))._1
}
object UnliftedModule {
  implicit def trainingMode[A, B, C, D, M <: StatefulModule2[A, B, C, D]](
      implicit t: TrainingMode[M],
      is: InitState[M, C]
  ) =
    TrainingMode.make[UnliftedModule[A, B, C, D, M]](
      m => UnliftedModule[A, B, C, D, M](m.statefulModule.asEval),
      m => UnliftedModule[A, B, C, D, M](m.statefulModule.asTraining)
    )
  implicit def load[A, B, C, D, M <: StatefulModule2[A, B, C, D]: Load](
      implicit is: InitState[M, C]
  ) =
    Load.make[UnliftedModule[A, B, C, D, M]](m =>
      tensors => UnliftedModule[A, B, C, D, M](m.statefulModule.load(tensors))
    )
  implicit def initState[A, B, C, D, M <: StatefulModule2[A, B, C, D]](
      implicit is: InitState[M, C]
  ) =
    InitState.make[UnliftedModule[A, B, C, D, M], Unit](m =>
      is.initState(m.statefulModule)
    )
}
trait GenericModule[A, B] extends (A => B) {
  def forward(x: A): B
  def apply(a: A): B = forward(a)
  def state: Seq[(Variable, PTag)]
  final def parameters =
    state.filter(v => v._1.needsGrad && v._2.updateDuringOptimization)
  final def gradients(
      loss: Variable,
      zeroGrad: Boolean = true
  ): Seq[Option[Tensor]] = {
    if (zeroGrad) {
      parameters.foreach {
        case (param, _) =>
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
case object NoTag extends LeafTag

trait TrainingMode[M] {
  def asEval(m: M): M
  def asTraining(m: M): M

}

object TrainingMode {
  def make[M](asEval1: M => M, asTraining1: M => M) = new TrainingMode[M] {
    def asEval(m: M) = asEval1(m)
    def asTraining(m: M) = asTraining1(m)
  }
  def identity[M] =
    TrainingMode.make(scala.Predef.identity[M], scala.Predef.identity[M])
}
trait Load[M] {
  def load(m: M, tensors: Seq[Tensor]): M
}
object Load {
  def identity[M]: Load[M] = Load.make[M](fun => _ => fun)
  def make[M](f: M => Seq[Tensor] => M) = new Load[M] {
    def load(m: M, tensors: Seq[Tensor]): M = f(m)(tensors)
  }
}
trait InitState[M, C] {
  def initState(m: M): C
}
object InitState {
  def make[M, C](f: M => C) = new InitState[M, C] {
    def initState(m: M) = f(m)
  }
}

case class MappedState[A, B, C, D, M <: StatefulModule[A, B, C]](
    statefulModule: M with StatefulModule[A, B, C],
    map: C => D
) extends StatefulModule2[A, B, C, D] {
  def state = statefulModule.state
  def forward(x: (A, C)) = {
    val (b, c) = statefulModule.forward(x)
    (b, map(c))
  }
}
object MappedState {
  implicit def trainingMode[A, B, C, D, M <: StatefulModule[A, B, C]](
      implicit t: TrainingMode[M]
  ) =
    TrainingMode.make[MappedState[A, B, C, D, M]](
      m => MappedState[A, B, C, D, M](m.statefulModule.asEval, m.map),
      m => MappedState[A, B, C, D, M](m.statefulModule.asTraining, m.map)
    )
  implicit def load[A, B, C, D, M <: StatefulModule[A, B, C]: Load] =
    Load.make[MappedState[A, B, C, D, M]](m =>
      tensors =>
        MappedState[A, B, C, D, M](m.statefulModule.load(tensors), m.map)
    )
  implicit def initState[A, B, C, D, M <: StatefulModule[A, B, C]](
      implicit is: InitState[M, C]
  ) =
    InitState.make[MappedState[A, B, C, D, M], C](m =>
      is.initState(m.statefulModule)
    )
}
