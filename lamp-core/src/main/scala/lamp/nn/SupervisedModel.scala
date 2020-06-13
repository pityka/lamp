package lamp.nn
import lamp.autograd.const
import aten.Tensor
import lamp.autograd.Variable
import lamp.autograd.TensorHelpers
import aten.ATen
import cats.effect.Resource
import cats.effect.IO

case class SupervisedModel(
    module: Module,
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
      val output = module.forward(const(samples))
      val loss = lossFunction(output, target)
      val lossAsDouble = TensorHelpers.toMat(loss.value).raw(0)
      val outputCloned = ATen.clone(output.value)
      loss.releaseAll
      (lossAsDouble, outputCloned)
    })(release.tupled)
  }
  def lossAndGradients(
      samples: Tensor,
      target: Tensor
  ): (Double, Seq[Option[Tensor]]) = {
    val output = module.forward(const(samples))
    val loss = lossFunction(output, target)
    val lossAsDouble = TensorHelpers.toMat(loss.value).raw(0)

    val gradients = module.gradients(loss)
    (lossAsDouble, gradients)
  }
  def lossAndGradientsAndOutput(
      samples: Tensor,
      target: Tensor
  ): Resource[IO, (Double, Seq[Option[Tensor]], Tensor)] = {
    val release = (_: Double, _: Seq[Option[Tensor]], outputCloned: Tensor) =>
      IO(outputCloned.release)
    Resource.make(IO {
      val output = module.forward(const(samples))
      val loss = lossFunction(output, target)
      val lossAsDouble = TensorHelpers.toMat(loss.value).raw(0)
      val outputCloned = ATen.clone(output.value)
      val gradients = module.gradients(loss)
      (lossAsDouble, gradients, outputCloned)
    })(release.tupled)
  }
  def zipOptimizer(optimizerFactory: Seq[(Tensor, PTag)] => Optimizer) =
    ModelWithOptimizer(
      this,
      optimizerFactory(module.parameters.map(v => (v._1.value, v._2)))
    )
}

case class ModelWithOptimizer(
    model: SupervisedModel,
    optimizer: Optimizer
)
