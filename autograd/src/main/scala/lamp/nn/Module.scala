package lamp.nn

import aten.Tensor
import lamp.autograd._
import aten.ATen
import aten.TensorOptions

object TrainLoop {
  def simple(
      module: Module,
      data1: Tensor,
      target1: Tensor,
      optimizerFactory: Seq[Variable] => Optimizer,
      epochs: Int
  ) = {
    val optim = optimizerFactory(module.parameters)
    val data: Variable = const(data1)
    val target: Variable = const(target1)

    var i = 0
    while (i < epochs) {
      val output = module.forward(data)
      val loss: Variable = (output - target).pow(2d).sum
      val gradients = module.gradients(loss)
      optim.step(gradients)
      i += 1
    }
  }
}

case class Sequential(members: Module*) extends Module {
  def parameters = members.flatMap(_.parameters)
  def forward(x: Variable) =
    members.foldLeft(x)((x, b) => b.forward(x))
}

case class FunctionModule(fun: Variable => Variable) extends Module {
  def parameters = Nil
  def forward(x: Variable): Variable = fun(x)
}

trait Module {
  def forward(x: Variable): Variable
  def parameters: Seq[Variable]
  def gradients(loss: Variable): Seq[Tensor] = {
    parameters.foreach(_.zeroGrad())
    loss.backprop()
    val g = parameters.map(_.partialDerivative.get)
    loss.release
    g
  }
}

trait Optimizer {
  def step(gradients: Seq[Tensor]): Unit
}

case class SGD(learningRate: Double, parameters: Seq[Tensor])
    extends Optimizer {
  def step(gradients: Seq[Tensor]) = {
    parameters.zip(gradients).foreach {
      case (param, gradients) =>
        ATen.add_out(param, param, gradients, learningRate)
    }
  }
}
