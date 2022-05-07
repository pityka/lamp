package lamp
import lamp.autograd.Variable
import lamp.autograd.{Constant, param}

/** Provides building blocks for neural networks
  *
  * Notable types:
  *   - [[nn.GenericModule]] is an abstraction on parametric functions
  *   - [[nn.Optimizer]] is an abstraction of gradient based optimizers
  *   - [[nn.LossFunction]] is an abstraction of loss functions, see the
  *     companion object for the implemented losses
  *   - [[nn.SupervisedModel]] combines a module with a loss
  *
  * Optimizers:
  *   - [[nn.AdamW]]
  *   - [[nn.SGDW]]
  *   - [[nn.RAdam]]
  *   - [[nn.Yogi]]
  *
  * Modules facilitating composing other modules:
  *   - [[nn.Sequential]] composes a homogenous list of modules (analogous to
  *     List)
  *   - [[nn.sequence]] composes a heterogeneous list of modules (analogous to
  *     tuples)
  *   - [[nn.EitherModule]] composes two modules in a scala.Either
  *
  * Examples of neural network building blocks, layers etc:
  *   - [[nn.Linear]] implements `W X + b` with parameters `W` and `b` and input
  *     `X`
  *   - [[nn.BatchNorm]], [[nn.LayerNorm]] implement batch and layer
  *     normalization
  *   - [[nn.MLP]] is a factory of a multilayer perceptron architecture
  */
package object nn {
  type Module = GenericModule[Variable, Variable]
  type StatefulModule[A, B, C] = GenericModule[(A, C), (B, C)]
  type StatefulModule2[A, B, C, D] = GenericModule[(A, C), (B, D)]

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

    val norm = Scope.leak { implicit scope =>
      STen
        .stack(
          gradients
            .map {
              case Some(g) =>
                Some(g.pow(2d).sum)
              case None => None
            }
            .collect { case Some(x) => x },
          0
        )
        .sum
        .sqrt
        .toDoubleArray(0)
    }
    if (norm > theta) {
      Scope.root { implicit scope =>
        val scalar = STen.scalarDouble(
          theta / norm,
          gradients.find(_.isDefined).get.get.options
        )
        gradients.foreach {
          case None =>
          case Some(g) =>
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

  private def ssz(ts: GenericModule[_, _]*) = ts.map(_.state.size).sum

  def loadMultiple[T1 <: GenericModule[_, _]: Load, T2 <: GenericModule[
    _,
    _
  ]: Load](
      t1: T1,
      t2: T2,
      tensors: Seq[STen]
  ) = {
    t1.load(tensors.take(ssz(t1)))
    t2.load(tensors.drop(ssz(t1)).take(ssz(t2)))
  }
  def loadMultiple[
      T1 <: GenericModule[_, _]: Load,
      T2 <: GenericModule[_, _]: Load,
      T3 <: GenericModule[_, _]: Load
  ](
      t1: T1,
      t2: T2,
      t3: T3,
      tensors: Seq[STen]
  ) = {
    t1.load(tensors.take(ssz(t1)))
    t2.load(tensors.drop(ssz(t1)).take(ssz(t2)))
    t3.load(tensors.drop(ssz(t1, t2)).take(ssz(t3)))
  }
  def loadMultiple[
      T1 <: GenericModule[_, _]: Load,
      T2 <: GenericModule[_, _]: Load,
      T3 <: GenericModule[_, _]: Load,
      T4 <: GenericModule[_, _]: Load
  ](
      t1: T1,
      t2: T2,
      t3: T3,
      t4: T4,
      tensors: Seq[STen]
  ) = {
    t1.load(tensors.take(ssz(t1)))
    t2.load(tensors.drop(ssz(t1)).take(ssz(t2)))
    t3.load(tensors.drop(ssz(t1, t2)).take(ssz(t3)))
    t4.load(tensors.drop(ssz(t1, t2, t3)).take(ssz(t4)))
  }
  def loadMultiple[
      T1 <: GenericModule[_, _]: Load,
      T2 <: GenericModule[_, _]: Load,
      T3 <: GenericModule[_, _]: Load,
      T4 <: GenericModule[_, _]: Load,
      T5 <: GenericModule[_, _]: Load
  ](
      t1: T1,
      t2: T2,
      t3: T3,
      t4: T4,
      t5: T5,
      tensors: Seq[STen]
  ) = {
    t1.load(tensors.take(ssz(t1)))
    t2.load(tensors.drop(ssz(t1)).take(ssz(t2)))
    t3.load(tensors.drop(ssz(t1, t2)).take(ssz(t3)))
    t4.load(tensors.drop(ssz(t1, t2, t3)).take(ssz(t4)))
    t5.load(tensors.drop(ssz(t1, t2, t3, t4)).take(ssz(t5)))
  }
  def loadMultiple[
      T1 <: GenericModule[_, _]: Load,
      T2 <: GenericModule[_, _]: Load,
      T3 <: GenericModule[_, _]: Load,
      T4 <: GenericModule[_, _]: Load,
      T5 <: GenericModule[_, _]: Load,
      T6 <: GenericModule[_, _]: Load
  ](
      t1: T1,
      t2: T2,
      t3: T3,
      t4: T4,
      t5: T5,
      t6: T6,
      tensors: Seq[STen]
  ) = {
    t1.load(tensors.take(ssz(t1)))
    t2.load(tensors.drop(ssz(t1)).take(ssz(t2)))
    t3.load(tensors.drop(ssz(t1, t2)).take(ssz(t3)))
    t4.load(tensors.drop(ssz(t1, t2, t3)).take(ssz(t4)))
    t5.load(tensors.drop(ssz(t1, t2, t3, t4)).take(ssz(t5)))
    t6.load(tensors.drop(ssz(t1, t2, t3, t4, t5)).take(ssz(t6)))
  }
  def loadMultiple[
      T1 <: GenericModule[_, _]: Load,
      T2 <: GenericModule[_, _]: Load,
      T3 <: GenericModule[_, _]: Load,
      T4 <: GenericModule[_, _]: Load,
      T5 <: GenericModule[_, _]: Load,
      T6 <: GenericModule[_, _]: Load,
      T7 <: GenericModule[_, _]: Load
  ](
      t1: T1,
      t2: T2,
      t3: T3,
      t4: T4,
      t5: T5,
      t6: T6,
      t7: T7,
      tensors: Seq[STen]
  ) = {
    t1.load(tensors.take(ssz(t1)))
    t2.load(tensors.drop(ssz(t1)).take(ssz(t2)))
    t3.load(tensors.drop(ssz(t1, t2)).take(ssz(t3)))
    t4.load(tensors.drop(ssz(t1, t2, t3)).take(ssz(t4)))
    t5.load(tensors.drop(ssz(t1, t2, t3, t4)).take(ssz(t5)))
    t6.load(tensors.drop(ssz(t1, t2, t3, t4, t5)).take(ssz(t6)))
    t7.load(tensors.drop(ssz(t1, t2, t3, t4, t5, t6)).take(ssz(t7)))
  }
  def loadMultiple[
      T1 <: GenericModule[_, _]: Load,
      T2 <: GenericModule[_, _]: Load,
      T3 <: GenericModule[_, _]: Load,
      T4 <: GenericModule[_, _]: Load,
      T5 <: GenericModule[_, _]: Load,
      T6 <: GenericModule[_, _]: Load,
      T7 <: GenericModule[_, _]: Load,
      T8 <: GenericModule[_, _]: Load
  ](
      t1: T1,
      t2: T2,
      t3: T3,
      t4: T4,
      t5: T5,
      t6: T6,
      t7: T7,
      t8: T8,
      tensors: Seq[STen]
  ) = {
    t1.load(tensors.take(ssz(t1)))
    t2.load(tensors.drop(ssz(t1)).take(ssz(t2)))
    t3.load(tensors.drop(ssz(t1, t2)).take(ssz(t3)))
    t4.load(tensors.drop(ssz(t1, t2, t3)).take(ssz(t4)))
    t5.load(tensors.drop(ssz(t1, t2, t3, t4)).take(ssz(t5)))
    t6.load(tensors.drop(ssz(t1, t2, t3, t4, t5)).take(ssz(t6)))
    t7.load(tensors.drop(ssz(t1, t2, t3, t4, t5, t6)).take(ssz(t7)))
    t8.load(tensors.drop(ssz(t1, t2, t3, t4, t5, t6, t7)).take(ssz(t8)))
  }
  def loadMultiple[
      T1 <: GenericModule[_, _]: Load,
      T2 <: GenericModule[_, _]: Load,
      T3 <: GenericModule[_, _]: Load,
      T4 <: GenericModule[_, _]: Load,
      T5 <: GenericModule[_, _]: Load,
      T6 <: GenericModule[_, _]: Load,
      T7 <: GenericModule[_, _]: Load,
      T8 <: GenericModule[_, _]: Load,
      T9 <: GenericModule[_, _]: Load
  ](
      t1: T1,
      t2: T2,
      t3: T3,
      t4: T4,
      t5: T5,
      t6: T6,
      t7: T7,
      t8: T8,
      t9: T9,
      tensors: Seq[STen]
  ) = {
    t1.load(tensors.take(ssz(t1)))
    t2.load(tensors.drop(ssz(t1)).take(ssz(t2)))
    t3.load(tensors.drop(ssz(t1, t2)).take(ssz(t3)))
    t4.load(tensors.drop(ssz(t1, t2, t3)).take(ssz(t4)))
    t5.load(tensors.drop(ssz(t1, t2, t3, t4)).take(ssz(t5)))
    t6.load(tensors.drop(ssz(t1, t2, t3, t4, t5)).take(ssz(t6)))
    t7.load(tensors.drop(ssz(t1, t2, t3, t4, t5, t6)).take(ssz(t7)))
    t8.load(tensors.drop(ssz(t1, t2, t3, t4, t5, t6, t7)).take(ssz(t8)))
    t9.load(tensors.drop(ssz(t1, t2, t3, t4, t5, t6, t7, t8)).take(ssz(t9)))
  }
  def loadMultiple[
      T1 <: GenericModule[_, _]: Load,
      T2 <: GenericModule[_, _]: Load,
      T3 <: GenericModule[_, _]: Load,
      T4 <: GenericModule[_, _]: Load,
      T5 <: GenericModule[_, _]: Load,
      T6 <: GenericModule[_, _]: Load,
      T7 <: GenericModule[_, _]: Load,
      T8 <: GenericModule[_, _]: Load,
      T9 <: GenericModule[_, _]: Load,
      T10 <: GenericModule[_, _]: Load
  ](
      t1: T1,
      t2: T2,
      t3: T3,
      t4: T4,
      t5: T5,
      t6: T6,
      t7: T7,
      t8: T8,
      t9: T9,
      t10: T10,
      tensors: Seq[STen]
  ) = {
    t1.load(tensors.take(ssz(t1)))
    t2.load(tensors.drop(ssz(t1)).take(ssz(t2)))
    t3.load(tensors.drop(ssz(t1, t2)).take(ssz(t3)))
    t4.load(tensors.drop(ssz(t1, t2, t3)).take(ssz(t4)))
    t5.load(tensors.drop(ssz(t1, t2, t3, t4)).take(ssz(t5)))
    t6.load(tensors.drop(ssz(t1, t2, t3, t4, t5)).take(ssz(t6)))
    t7.load(tensors.drop(ssz(t1, t2, t3, t4, t5, t6)).take(ssz(t7)))
    t8.load(tensors.drop(ssz(t1, t2, t3, t4, t5, t6, t7)).take(ssz(t8)))
    t9.load(tensors.drop(ssz(t1, t2, t3, t4, t5, t6, t7, t8)).take(ssz(t9)))
    t10.load(
      tensors.drop(ssz(t1, t2, t3, t4, t5, t6, t7, t8, t9)).take(ssz(t10))
    )
  }
  def loadMultiple[
      T1 <: GenericModule[_, _]: Load,
      T2 <: GenericModule[_, _]: Load,
      T3 <: GenericModule[_, _]: Load,
      T4 <: GenericModule[_, _]: Load,
      T5 <: GenericModule[_, _]: Load,
      T6 <: GenericModule[_, _]: Load,
      T7 <: GenericModule[_, _]: Load,
      T8 <: GenericModule[_, _]: Load,
      T9 <: GenericModule[_, _]: Load,
      T10 <: GenericModule[_, _]: Load,
      T11 <: GenericModule[_, _]: Load
  ](
      t1: T1,
      t2: T2,
      t3: T3,
      t4: T4,
      t5: T5,
      t6: T6,
      t7: T7,
      t8: T8,
      t9: T9,
      t10: T10,
      t11: T11,
      tensors: Seq[STen]
  ) = {
    t1.load(tensors.take(ssz(t1)))
    t2.load(tensors.drop(ssz(t1)).take(ssz(t2)))
    t3.load(tensors.drop(ssz(t1, t2)).take(ssz(t3)))
    t4.load(tensors.drop(ssz(t1, t2, t3)).take(ssz(t4)))
    t5.load(tensors.drop(ssz(t1, t2, t3, t4)).take(ssz(t5)))
    t6.load(tensors.drop(ssz(t1, t2, t3, t4, t5)).take(ssz(t6)))
    t7.load(tensors.drop(ssz(t1, t2, t3, t4, t5, t6)).take(ssz(t7)))
    t8.load(tensors.drop(ssz(t1, t2, t3, t4, t5, t6, t7)).take(ssz(t8)))
    t9.load(tensors.drop(ssz(t1, t2, t3, t4, t5, t6, t7, t8)).take(ssz(t9)))
    t10.load(
      tensors.drop(ssz(t1, t2, t3, t4, t5, t6, t7, t8, t9)).take(ssz(t10))
    )
    t11.load(
      tensors.drop(ssz(t1, t2, t3, t4, t5, t6, t7, t8, t9, t10)).take(ssz(t11))
    )
  }
  def loadMultiple[
      T1 <: GenericModule[_, _]: Load,
      T2 <: GenericModule[_, _]: Load,
      T3 <: GenericModule[_, _]: Load,
      T4 <: GenericModule[_, _]: Load,
      T5 <: GenericModule[_, _]: Load,
      T6 <: GenericModule[_, _]: Load,
      T7 <: GenericModule[_, _]: Load,
      T8 <: GenericModule[_, _]: Load,
      T9 <: GenericModule[_, _]: Load,
      T10 <: GenericModule[_, _]: Load,
      T11 <: GenericModule[_, _]: Load,
      T12 <: GenericModule[_, _]: Load
  ](
      t1: T1,
      t2: T2,
      t3: T3,
      t4: T4,
      t5: T5,
      t6: T6,
      t7: T7,
      t8: T8,
      t9: T9,
      t10: T10,
      t11: T11,
      t12: T12,
      tensors: Seq[STen]
  ) = {
    t1.load(tensors.take(ssz(t1)))
    t2.load(tensors.drop(ssz(t1)).take(ssz(t2)))
    t3.load(tensors.drop(ssz(t1, t2)).take(ssz(t3)))
    t4.load(tensors.drop(ssz(t1, t2, t3)).take(ssz(t4)))
    t5.load(tensors.drop(ssz(t1, t2, t3, t4)).take(ssz(t5)))
    t6.load(tensors.drop(ssz(t1, t2, t3, t4, t5)).take(ssz(t6)))
    t7.load(tensors.drop(ssz(t1, t2, t3, t4, t5, t6)).take(ssz(t7)))
    t8.load(tensors.drop(ssz(t1, t2, t3, t4, t5, t6, t7)).take(ssz(t8)))
    t9.load(tensors.drop(ssz(t1, t2, t3, t4, t5, t6, t7, t8)).take(ssz(t9)))
    t10.load(
      tensors.drop(ssz(t1, t2, t3, t4, t5, t6, t7, t8, t9)).take(ssz(t10))
    )
    t11.load(
      tensors.drop(ssz(t1, t2, t3, t4, t5, t6, t7, t8, t9, t10)).take(ssz(t11))
    )
    t12.load(
      tensors
        .drop(ssz(t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11))
        .take(ssz(t12))
    )
  }

}
