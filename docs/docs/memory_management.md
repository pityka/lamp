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
Autograd variabels own up to two tensors: their value and optionally their partial derivatives.

`lamp.STen` is a shallow wrapper around `aten.Tensor`s. It ensures that an appropriate scope is present
before allocation.

Example:

``` scala mdoc
  def squaredEuclideanDistance(v1: STen, v2: STen)(
      implicit scope: Scope // parent scope
  ): STen = {
    Scope { implicit scope => // this is a local scope cleared up when block ends
      val outer = v1.mm(v2.t) // these allocations will get released at the end of the block
      val n1 = (v1 * v1).rowSum
      val n2 = (v2 * v2).rowSum
      (n1 + n2.t - outer * 2) // the return value of the block is not released, but moved to the parent scope
    }
  }
```


