package lamp.nn
import aten.Tensor
import lamp.autograd.Variable
import aten.ATen
import cats.effect.Resource
import cats.effect.IO
import lamp.syntax

case class SupervisedModel[I, M <: GenericModule[I, Variable]](
    module: M with GenericModule[I, Variable],
    lossFunction: LossFunction
)(implicit tm: TrainingMode[M]) {
  def asEval = copy(module = module.asEval)
  def asTraining = copy(module = module.asTraining)
  def lossAndOutput(
      samples: I,
      target: Tensor
  ): Resource[IO, (Double, Tensor, Long)] = {
    val release = (_: Double, outputCloned: Tensor, _: Long) =>
      IO(outputCloned.release)
    Resource.make(IO {
      val output = module.forward(samples)
      val (loss, examples) = lossFunction(output, target)
      val lossAsDouble = loss.value.toMat.raw(0)
      val outputCloned = ATen.clone(output.value)
      loss.releaseAll
      (lossAsDouble, outputCloned, examples)
    })(release.tupled)
  }
  def lossAndGradients(
      samples: I,
      target: Tensor
  ): (Double, Seq[Option[Tensor]]) = {
    val output = module.forward(samples)
    val (loss, _) = lossFunction(output, target)
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

case class ModelWithOptimizer[I, M <: GenericModule[I, Variable]](
    model: SupervisedModel[I, M],
    optimizer: Optimizer
) {
  def release() = {
    optimizer.release
    model.module.state.foreach(_._1.value.release)
  }
}
