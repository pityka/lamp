package lamp

/** Implements reverse mode automatic differentiaton
  *
  * The main types in this package are [[lamp.autograd.Variable]] and [[lamp.autograd.Op]].
  * The computational graph built by this package consists of vertices representing values (as [[lamp.autograd.Variable]]) and vertices representing operations (as [[lamp.autograd.Op]]).
  *
  * Variables contain the value of a `R^n^ => R^m^` function.
  * Variables may also contain the partial derivative of their argument with respect to a single scalar.
  * A Variable whose value is a scalar (m=1) can trigger the computation of partial derivatives of all
  * the intermediate upstream Variables.
  * Computing partial derivatives with respect to non-scalar variables is not supported.
  *
  * A constant Variable  may be created with the `const`
  * or `param` factory method in this package. `const` may be used for constants which
  * do not need their partial derivatives to be computed. `param` on the other hand
  * create Variables which will fill in their partial derivatives. Further variables may be created by the
  * methods in this class, eventually expressing more complex `R^n^ => R^m^` functions.
  * ===Example===
  * {{{
  * lamp.Scope.root{ implicit scope =>
  *   // x is constant (depends on no other variables) and won't compute a partial derivative
  *   val x = lamp.autograd.const(STen.eye(3, STenOptions.d))
  *   // y is constant but will compute a partial derivative
  *   val y = lamp.autograd.param(STen.ones(List(3,3), STenOptions.d))
  *
  *   // z is a Variable with x and y dependencies
  *   val z = x+y
  *
  *   // w is a Variable with z as a direct and x, y as transient dependencies
  *   val w = z.sum
  *   // w is a scalar (number of elements is 1), thus we can call backprop() on it.
  *   // calling backprop will fill out the partial derivatives of the upstream variables
  *   w.backprop()
  *
  *   // partialDerivative is empty since we created `x` with `const`
  *   assert(x.partialDerivative.isEmpty)
  *
  *   // `y`'s partial derivatie is defined and is computed
  *   // it holds `y`'s partial derivative with respect to `w`, the scalar which we called backprop() on
  *   assert(y.partialDerivative.isDefined)
  *
  * }
  * }}}
  *
  * This package may be used to compute the derivative of any function, provided the function can
  * be composed out of the provided methods. A particular use case is gradient based optimization.
  *
  * @see [[https://arxiv.org/pdf/1811.05031.pdf]] for a review of the algorithm
  * @see [[lamp.autograd.Op]] for how to implement a new operation
  */
package object autograd {

  type GC[_] = GraphConfiguration

  def withDefaultGraphConfig[T](f: GraphConfiguration => T): T = f(
    implicits.defaultGraphConfiguration
  )

  def graphConfiguration(implicit conf: GraphConfiguration) = conf

  object implicits {
    implicit val defaultGraphConfiguration =
      GraphConfiguration(downCastEnabled = false)
  }

  def const(m: STen): Constant =
    ConstantWithoutGrad(m)

  def const(
      m: Double,
      tOpt: STenOptions = STenOptions.d
  )(implicit scope: Scope): Constant =
    ConstantWithoutGrad(STen.scalarDouble(m, tOpt))

  def param(m: STen)(implicit scope: Scope): ConstantWithGrad =
    ConstantWithGrad(m, STen.zerosLike(m)(scope))
  def param(
      m: Double,
      tOpt: STenOptions = STenOptions.d
  )(implicit scope: Scope): ConstantWithGrad =
    Scope { implicit scope =>
      val scalar = STen.scalarDouble(m, tOpt).view(1)
      ConstantWithGrad(scalar, STen.zerosLike(scalar)(scope))
    }

  private[lamp] def measure[T](tag: String)(body: => T): T = {
    val t1 = System.nanoTime()
    val r = body
    val t2 = System.nanoTime()
    println(s"$tag" + (t2 - t1) * 1e-9)
    r
  }

  private[lamp] def atLeastFloat[S: Sc](op: STenOptions) = {
    if (op.scalarTypeByte == HalfPrecision.scalarTypeByte) op.toFloat else op
  }
  private[lamp] def castUp[S: Sc](v: STen) = if (
    v.scalarTypeByte == HalfPrecision.scalarTypeByte
  ) v.castToFloat
  else v
  private[lamp] def castDown[S: Sc, G: GC](s: STen): STen =
    if (implicitly[GraphConfiguration].downCastEnabled) s.castToHalf else s
  private[lamp] def castUp[S: Sc](vs: Seq[STen]): Seq[STen] = vs.map(castUp)
  private[lamp] def castUpOnCPU[S: Sc](v: STen) = if (
    v.isCPU && v.scalarTypeByte == HalfPrecision.scalarTypeByte
  ) v.castToFloat
  else v

  implicit class CastSyntax(v: STen) {
    def castDown[S: Sc, G: GC] = lamp.autograd.castDown(v)
    def castUp[S: Sc] = lamp.autograd.castUp(v)
    def castUpOnCPU[S: Sc] = lamp.autograd.castUpOnCPU(v)

  }
  implicit class CastSyntaxSeq(v: Seq[STen]) {
    def castUp[S: Sc] = lamp.autograd.castUp(v)
  }

}
