---
title: 'Defining modules'
weight: 4
---

Similar to PyTorch, Lamp organizes neural networks into modules. 
A module holds a state (typically parameters) and provide a function from some type A to type B while potentially closing over its state.
A simple interface could be: 
```scala mdoc:compile-only
import lamp.autograd.Variable
trait Module[A, B] {
  def forward(x: A): B
  val state: Seq[Variable]
}
```

Lamp provides generic building blocks to combine modules (sequences, Either, Option).
The most common combination of modules is to sequence them, that is function composition.
This one can do with the apply methods in the `lamp.nn.sequence` object.

As an example here is the complete source code of the linear layer:

```scala mdoc:compile-only
import lamp.autograd.Variable
import lamp.nn.Module
import lamp.nn.Linear
import lamp.Sc

case class LinearDemo(weights: Variable, bias: Option[Variable]) extends Module {

  override val state = List(
    weights -> Linear.Weights
  ) ++ bias.toList.map(b => (b, Linear.Bias))

  def forward[S:Sc](x: Variable): Variable = {
    val v = x.mm(weights.t)
    bias.map(_ + v).getOrElse(v)

  }
}
```
This module has two state variables `wight` and `bias`. 
The `state` field takes these values along with a small tag object which helps identifying the parameter during runtime.
The `forward` method applies a linear to the input variable. The `forward` method also takes an implicit `Scope` instance since the return type of `forward` is `Variable` and Variables are only constructable with a `Scope` instance. 

Additional features around modules are added with type classes: putting a module into training or evaluation mode needs some compile time introspection and this is added with the `TrainingMode` type class. Loading a module from a sequence of Tensors also needs compile time introspection and is added with the `Load` type class.

Some transformations in a neural network are stateless, e.g. applying a nonlinearity: they only depend on the input but have no parameters. These can be written without a specific module with `Fun`

```scala mdoc:compile-only
lamp.nn.Fun(implicit scope => _.relu)
```