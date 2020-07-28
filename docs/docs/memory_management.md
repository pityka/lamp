---
title: 'Memory management'
weight: 3
---

Lamp allocates data as ATen tensors which are stored off heap. 
Native ATen tensors are exposed to the JVM via the `aten.Tensor` class. 
Each `aten.Tensor` JVM object is a handle to the tensor - actually handle to native `Tensor` object which is a handle itself to the tensor's data. 

Tensors must be released manually with the `aten.Tensor#release` or `releaseAll` methods. A double release might crash the VM.

# autograd Variables

In contrast with `aten.Tensor`s `lamp.autograd.Variable`s are managed. Allocation of these are recorded into a pool (`lamp.autograd.AllocatedVariablePool`) and the pool is released at certain points - after each backpropagation step - during the training loop.

The `const(t:Tensor)` or `param(t:Tensor)` Variable factories do not append their argument to the pool, therefore those tensors are not released when the pool is cleared. Use the `releasable` method on a variable to ensure the pool will release their tensor.