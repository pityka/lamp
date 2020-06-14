package lamp.nn
import lamp.autograd.const
import aten.Tensor
import lamp.autograd.Variable
import lamp.autograd.TensorHelpers
import aten.ATen
import cats.effect.Resource
import cats.effect.IO
import cats.data.State
import lamp.syntax

case class SupervisedModel[T](
    module: StatefulModule[T],
    initState: T,
    lossFunction: (Variable, Tensor) => Variable
) {
  def asEval = copy(module = module.asEval)
  def asTraining = copy(module = module.asTraining)
  def lossAndOutput(
      samples: Tensor,
      target: Tensor
  ): Resource[IO, (Double, Tensor)] = {
    val release = (_: Double, outputCloned: Tensor) => IO(outputCloned.release)
    Resource.make(IO {
      val (output, _) = module.forward1(const(samples), initState)
      val loss = lossFunction(output, target)
      val lossAsDouble = loss.value.toMat.raw(0)
      val outputCloned = ATen.clone(output.value)
      loss.releaseAll
      (lossAsDouble, outputCloned)
    })(release.tupled)
  }
  def lossAndGradients(
      samples: Tensor,
      target: Tensor
  ): (Double, Seq[Option[Tensor]]) = {
    val (output, _) = module.forward1(const(samples), initState)
    val loss = lossFunction(output, target)
    val lossAsDouble = loss.value.toMat.raw(0)

    val gradients = module.gradients(loss)
    (lossAsDouble, gradients)
  }

  def zipOptimizer(optimizerFactory: Seq[(Tensor, PTag)] => Optimizer) =
    ModelWithOptimizer(
      this,
      optimizerFactory(module.parameters.map(v => (v._1.value, v._2)))
    )
}

case class ModelWithOptimizer[St](
    model: SupervisedModel[St],
    optimizer: Optimizer
)
