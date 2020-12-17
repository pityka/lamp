package lamp
package object autograd {

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

  def measure[T](tag: String)(body: => T): T = {
    val t1 = System.nanoTime()
    val r = body
    val t2 = System.nanoTime()
    println(s"$tag" + (t2 - t1) * 1e-9)
    r
  }
}
