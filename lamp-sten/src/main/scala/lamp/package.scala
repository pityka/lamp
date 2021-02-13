import aten.Tensor

/** Lamp provides utilities to build state of the art machine learning applications
  *
  * =Overview=
  * Notable types and packages:
  *
  *  - [[lamp.STen]] is a memory managed wrapper around aten.ATen, an off the heap, native n-dimensionl array backed by libtorch.
  *  - [[lamp.autograd]] implements reverse mode automatic differentiation.
  *  - [[lamp.nn]] contains neural network building blocks, see e.g. [[lamp.nn.Linear]].
  *  - [[lamp.data.IOLoops]] implements a training loop and other data related abstractions.
  *  - [[lamp.knn]] implements k-nearest neighbor search on the CPU and GPU
  *  - [[lamp.umap.Umap]] implements the UMAP dimension reduction algorithm
  *  - [[lamp.onnx]] implements serialization of computation graphs into ONNX format
  *  - lamp.io contains CSV and NPY readers
  *
  * ===How to get data into lamp===
  * Use one of the file readers in lamp.io or one of the factories in [[lamp.STen$]].
  *
  * ===How to define a custom neural network layer===
  * See the documentation on [[lamp.nn.GenericModule]]
  *
  * ===How to compose neural network layers===
  * See the documentation on [[lamp.nn]]
  *
  * ===How to train models===
  * See the training loops in [[lamp.data.IOLoops]]
  *
  */
package object lamp {
  type Sc[_] = Scope

  def scope(implicit s: Scope) = s
  def scoped(r: Tensor)(implicit s: Scope): Tensor = s.apply(r)

}
