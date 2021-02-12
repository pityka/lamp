import aten.Tensor

/** Lamp provides utilities to build state of the art machine learning applications
  *
  * ==Overview==
  * Notable types and packages:
  *
  *  - [[lamp.STen]] is a memory managed wrapper around aten.ATen, an off the heap, native n-dimensionl array backed by libtorch.
  *  - [[lamp.autograd]] implements reverse mode automatic differentiation.
  *  - [[lamp.nn]] contains neural network building blocks, see e.g. [[lamp.nn.Linear]].
  *  - [[lamp.data.IOLoops]] implements a training loop and other data related abstractions.
  *
  */
package object lamp {
  type Sc[_] = Scope

  def scope(implicit s: Scope) = s
  def scoped(r: Tensor)(implicit s: Scope): Tensor = s.apply(r)

}
