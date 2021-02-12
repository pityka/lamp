package lamp.nn

import lamp.autograd._
import lamp.Sc
import lamp.Scope
import lamp.scope
import lamp.STen
import lamp.Movable

case class EitherModule[
    A,
    B,
    M1 <: GenericModule[A, B],
    M2 <: GenericModule[A, B]
](
    members: Either[M1 with GenericModule[A, B], M2 with GenericModule[A, B]]
) extends GenericModule[A, B] {
  override def state =
    members.fold(_.state, _.state)
  def forward[S: Sc](x: A) =
    members.fold(_.forward(x), _.forward(x))
}
object EitherModule {

  implicit def trainingMode[
      A,
      B,
      M1 <: GenericModule[A, B]: TrainingMode,
      M2 <: GenericModule[A, B]: TrainingMode
  ] =
    TrainingMode.make[EitherModule[A, B, M1, M2]](
      module =>
        EitherModule(module.members.left.map(_.asEval).right.map(_.asEval)),
      module =>
        EitherModule(
          module.members.left.map(_.asTraining).right.map(_.asTraining)
        )
    )

  implicit def load[
      A,
      B,
      M1 <: GenericModule[A, B]: Load,
      M2 <: GenericModule[A, B]: Load
  ] =
    Load.make[EitherModule[A, B, M1, M2]] { module => tensors =>
      module.members.fold(_.load(tensors), _.load(tensors))
    }

  case class Tag[T <: PTag](t: T, idx: Int) extends PTag {
    def leaf = t
  }
}

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
  def forward[S: Sc](x: A) =
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
      module.members.foldLeft((List[Unit](), tensors)) {
        case ((acc, params), member) =>
          val numParam = member.state.size
          val loaded = member.load(params.take(numParam))
          (acc.:+(loaded), params.drop(numParam))

      }
      ()
    }

  case class Tag[T <: PTag](t: T, idx: Int) extends PTag {
    def leaf = t
  }
}

case class Fun(fun: Scope => Variable => Variable) extends Module {
  def state = Nil
  def forward[S: Sc](x: Variable): Variable = fun(scope)(x)
}
object Fun {
  implicit val trainingMode = TrainingMode.identity[Fun]
  implicit val load = Load.identity[Fun]
}

case class GenericFun[A, B](fun: Scope => A => B) extends GenericModule[A, B] {
  def state = Nil
  def forward[S: Sc](x: A): B = fun(scope)(x)
}
object GenericFun {
  implicit def trainingMode[A, B] = TrainingMode.identity[GenericFun[A, B]]
  implicit def load[A, B] = Load.identity[GenericFun[A, B]]
}

case class LiftedModule[M <: Module](mod: M with Module)
    extends StatefulModule[Variable, Variable, Unit] {
  def state = mod.state
  def forward[S: Sc](x: (Variable, Unit)) = (mod.forward(x._1), ())
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
    Load.make[LiftedModule[M]](m => tensors => m.mod.load(tensors))
  implicit def initState[M <: Module] =
    InitState.make[LiftedModule[M], Unit](_ => ())
}

case class UnliftedModule[A, B, C, D, M <: StatefulModule2[A, B, C, D]](
    statefulModule: M with StatefulModule2[A, B, C, D]
)(implicit init: InitState[M, C])
    extends GenericModule[A, B] {
  def state = statefulModule.state
  def forward[S: Sc](x: A) =
    statefulModule.forward((x, statefulModule.initState))._1
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
  implicit def load[A, B, C, D, M <: StatefulModule2[A, B, C, D]: Load] =
    Load.make[UnliftedModule[A, B, C, D, M]](m =>
      tensors => m.statefulModule.load(tensors)
    )
  implicit def initState[A, B, C, D, M <: StatefulModule2[A, B, C, D]](
      implicit is: InitState[M, C]
  ) =
    InitState.make[UnliftedModule[A, B, C, D, M], Unit](m =>
      is.initState(m.statefulModule)
    )
}

object GenericModule {
  implicit def movable[A, B]: Movable[GenericModule[A, B]] =
    Movable.nonEmpty[GenericModule[A, B]] { m =>
      m.state
        .flatMap(_._1 match {
          case ConstantWithGrad(value, pd) => List(value.value, pd.value)
          case ConstantWithoutGrad(value)  => List(value.value)
        })
        .toList
    }
}

/** Base type of modules
  *
  * Modules are functions of type `(Seq[lamp.autograd.Constant],A) => B`, where
  * the `Seq[lamp.autograd.Constant]` arguments are optimizable parameters and `A` is a non-optimizable
  * input.
  *
  * Modules provide a way to build composite functions while also keep track of the parameter list of the
  * composite function.
  *
  * ===Example===
  * {{{
  * case object Weights extends LeafTag
  * case object Bias extends LeafTag
  * case class Linear(weights: Constant, bias: Option[Constant]) extends Module {
  *
  *  override val state = List(
  *    weights -> Weights
  *  ) ++ bias.toList.map(b => (b, Bias))
  *
  *  def forward[S: Sc](x: Variable): Variable = {
  *    val v = x.mm(weights)
  *    bias.map(_ + v).getOrElse(v)
  *
  *  }
  *}
  * }}}
  *
  * Some other attributes of modules are attached by type classes e.g. with the [[nn.TrainingMode]], [[nn.Load]]
  * type classes.
  *
  * @tparam A the argument type of the module
  * @tparam B the value type of the module
  * @see [[nn.Module]] is an alias for simple `Variable => Variable` modules
  */
trait GenericModule[A, B] {

  /** The implementation of the function.
    *
    * In addition of `x` it can also use all the `state to compute its value.
    *
    */
  def forward[S: Sc](x: A): B

  /** Alias of forward */
  def apply[S: Sc](a: A): B = forward(a)

  /** List of optimizable, or non-optimizable, but stateful parameters
    *
    * Stateful means that the state is carried over the repeated forward calls.
    */
  def state: Seq[(Constant, PTag)]

  /** Returns the state variables which need gradient computation. */
  final def parameters =
    state.filter(v => v._1.needsGrad)

  /** Computes the gradient of loss with respect to the parameters.  */
  final def gradients(
      loss: Variable,
      zeroGrad: Boolean = true
  ): Seq[Option[STen]] = {
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
    g
  }

  /** Returns the total number of optimizable parameters.  */
  final def learnableParameters =
    parameters.filter(_._1.needsGrad).map(_._1.value.numel).sum
}

/** A small trait to mark paramters for unique identification */
trait PTag {
  def leaf: PTag
}
object PTag {
  implicit def isMovable = Movable.empty[PTag]
}
trait LeafTag extends PTag {
  def leaf: PTag = this
}
case object NoTag extends LeafTag

/** Type class about how to switch a module into training or evaluation mode */
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

/** Type class about how to load the contents of the state of modules from external tensors */
trait Load[M] {
  def load(m: M, tensors: Seq[STen]): Unit
}
object Load {
  def identity[M]: Load[M] = Load.make[M](_ => _ => ())
  def make[M](f: M => Seq[STen] => Unit) = new Load[M] {
    def load(m: M, tensors: Seq[STen]): Unit = f(m)(tensors)
  }
}

/** Type class about how to initialize recurrent neural networks */
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
  def forward[S: Sc](x: (A, C)) = {
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
      tensors => m.statefulModule.load(tensors)
    )
  implicit def initState[A, B, C, D, M <: StatefulModule[A, B, C]](
      implicit is: InitState[M, C]
  ) =
    InitState.make[MappedState[A, B, C, D, M], C](m =>
      is.initState(m.statefulModule)
    )
}
