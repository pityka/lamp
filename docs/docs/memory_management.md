---
title: 'Memory management'
weight: 3
---

Lamp allocates data as ATen tensors which are stored off heap. 
Native ATen tensors are exposed to the JVM via the `aten.Tensor` class. 
Each `aten.Tensor` JVM object is a handle to the tensor - actually handle to native `Tensor` object which is a handle itself to the tensor's data. 

Tensors must be released manually with the `aten.Tensor#release` or `releaseAll` methods. A double release might crash the VM.

# autograd Variables and STen tensors

In contrast with `aten.Tensor`s `lamp.autograd.Variable`s and `lamp.STen`s are managed. Allocation of these require a scope (`lamp.Scope`) which demarkate the lifetime of the variable.
Autograd variables own up to two tensors: their value and optionally their partial derivatives.

`lamp.STen` is a shallow wrapper around `aten.Tensor`s. It ensures that an appropriate scope is present
before allocation and it provides a more fluent chainable API.

Example:

``` scala mdoc
  def squaredEuclideanDistance(v1: STen, v2: STen)(
      implicit scope: Scope // parent scope
  ): STen = {
    Scope { implicit scope => // this is a local scope cleared up when block ends
      val outer = v1.mm(v2.t) // these allocations will get released at the end of the block
      val n1 = (v1 * v1).rowSum
      val n2 = (v2 * v2).rowSum
      (n1 + n2.t - outer * 2) 
    } // once the block exits all resources allocated within the block are released, with the exception of the 
      // return value which is moved to the parent scope
  }
```

# `lamp.Scope`

Both `STen` and `Variable` own references to `aten.Tensor`s which need to be managed (released) manually. 
The constructors of STen and Variable take an instance of `Scope` and register the Tensors with the Scope. 
A Scope can be released which releases all the registered Tensors. 
This simplifies memory management because a `Scope` instance can be injected into a Scala lexical block and released once the block exits. 

A `Scope` may be built with any of the factory methods in its companion object: 

`Scope.apply` , `Scope.leak` and `Scope.root`: these factories take a lambda, thus inject the `Scope` instance in the lexical scope of the lambda. 

`Scope.apply` is meant to be used in scope which itself has a parent scope. It will not release the return value, but move it to its parent scope. Consequently the return type of the lambda it takes are restricted to members of the `Movable` type class:
```scala 
def apply[A: Movable](
      op: Scope => A
  )(implicit parent: Scope): A
```

The `Movable` type class provides compile time introspection so that the library can extract the list of Tensors from the return value and move them to the parent scope. 
It is defined as 
```scala 
trait Movable[-R] {
  def list(movable: R): List[Tensor]
}
```
Most regular Scala types and primities have a `Movable` instance which return the empty list.
`lamp.STen` is member of the `Movable` type class.
On the contrary autograd Variables are not movable because their partial derivatives usually depend on a chain of other varibles thus they are not simple returnable from a code block.

`Scope.root` is meant to be used as the outermost Scope. It can not return anything, thus it takes a lambda with a Unit return type.

`Scope.leak` is mean to be used when the return value of the scope is managed by the VM, that is any regular Scala object or primitive. It is named `leak` to warn the programmer for possible resource leaks. It is always possible to use the `Scope.apply` factory by defining the necessary trivial `Movable` instance with `Movable.empty[T]`.

For more complex control flows there is a `Scope.free` factory method which returns a free standing `Scope` instance not bound to any lexical scope. The programmer has to make sure to release it. 

The `Scope` companion object also has factories for cats effect Resources and brackets.


