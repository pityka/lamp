package lamp
import lamp.autograd.Variable
import aten.ATen
import lamp.util.syntax
import lamp.autograd.{Constant, param}

/** Provides building blocks for neural networks
  *
  * Notable types:
  *   - [[nn.GenericModule]] is an abstraction on parametric functions
  *   - [[nn.Optimizer]] is an abstraction of gradient based optimizers
  *   - [[nn.LossFunction]] is an abstraction of loss functions, see the companion object for the implemented losses
  *   - [[nn.SupervisedModel]] combines a module with a loss
  *
  * Optimizers:
  *   - [[nn.AdamW]]
  *   - [[nn.SGDW]]
  *   - [[nn.RAdam]]
  *   - [[nn.Yogi]]
  *
  * Modules facilitating composing other modules:
  *   - [[nn.Sequential]] composes a homogenous list of modules (analogous to List)
  *   - [[nn.sequence]] composes a heterogeneous list of modules (analogous to tuples)
  *   - [[nn.EitherModule]] composes two modules in a scala.Either
  *
  * Examples of neural network building blocks, layers etc:
  *   - [[nn.Linear]] implements `W X + b` with parameters `W` and `b` and input `X`
  *   - [[nn.BatchNorm]], [[nn.LayerNorm]] implement batch and layer normalization
  *   - [[nn.MLP]] is a factory of a multilayer perceptron architecture
  */
package object nn {
  type Module = GenericModule[Variable, Variable]
  type StatefulModule[A, B, C] = GenericModule[(A, C), (B, C)]
  type StatefulModule2[A, B, C, D] = GenericModule[(A, C), (B, D)]
  type GraphModule = GenericModule[(Variable, Variable), (Variable, Variable)]

  implicit class TrainingModeSyntax[M: TrainingMode](m: M) {
    def asEval: M = implicitly[TrainingMode[M]].asEval(m)
    def asTraining: M = implicitly[TrainingMode[M]].asTraining(m)
  }
  implicit class LoadSyntax[M: Load](m: M) {
    def load(tensors: Seq[STen]): Unit =
      implicitly[Load[M]].load(m, tensors)
  }
  implicit class InitStateSyntax[M, C](m: M)(implicit is: InitState[M, C]) {
    def initState = is.initState(m)
  }

  implicit class ToLift[M <: Module](mod: M with Module) {
    def lift = LiftedModule(mod)
  }
  implicit class ToUnlift[A, B, C, D, M <: StatefulModule2[A, B, C, D]](
      mod: M with StatefulModule2[A, B, C, D]
  )(implicit
      is: InitState[M, C]
  ) {
    def unlift = UnliftedModule(mod)(is)
  }
  implicit class ToMappedState[A, B, C, M <: StatefulModule[A, B, C]](
      mod: M with StatefulModule[A, B, C]
  ) {
    def mapState[D](f: C => D) = MappedState[A, B, C, D, M](mod, f)
  }
  implicit class ToWithInit[A, B, C, M <: StatefulModule[A, B, C]](
      mod: M with StatefulModule[A, B, C]
  ) {
    def withInit(c: C) = WithInit[A, B, C, M](mod, c)
  }

  def gradientClippingInPlace(
      gradients: Seq[Option[STen]],
      theta: Double
  ): Unit = {
    val norm = math.sqrt((gradients.map {
      case Some(g) =>
        val tmp = ATen.pow_0(g.value, 2d)
        val d = ATen.sum_0(tmp).toMat.raw(0)
        tmp.release
        d
      case None => 0d
    }: Seq[Double]).sum)
    if (norm > theta) {
      gradients.foreach {
        case None =>
        case Some(g) =>
          Scope.root { implicit scope =>
            val scalar = STen.scalarDouble(theta / norm, g.options)
            g *= scalar
          }

      }
    }
  }

  def initLinear[S: Sc](in: Int, out: Int, tOpt: STenOptions): Constant = param(
    STen.normal(
      0d,
      math.sqrt(2d / (out + in)),
      List(in, out),
      tOpt
    )
  )
}
