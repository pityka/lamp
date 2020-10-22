import aten.Tensor
package object lamp {
  type Sc[_] = Scope

  def scope(implicit s: Scope) = s
  def scoped(r: Tensor)(implicit s: Scope): Tensor = s.apply(r)

}
