package lamp.nn
import lamp.autograd.Variable
import cats.effect.Resource
import cats.effect.IO
import lamp.Scope
import lamp.STen

case class SupervisedModel[I, M <: GenericModule[I, Variable]](
    module: M with GenericModule[I, Variable],
    lossFunction: LossFunction
)(implicit tm: TrainingMode[M]) {
  def asEval = copy(module = module.asEval)
  def asTraining = copy(module = module.asTraining)
  def lossAndOutput(
      samples: I,
      target: STen
  ): Resource[IO, (Double, STen, Long)] = {

    Scope.inResource.map { implicit scope =>
      val output = module.forward(samples)
      val (loss, examples) = lossFunction(output, target)
      val lossAsDouble = loss.value.toMat.raw(0)
      val outputCloned = output.value.cloneTensor
      (lossAsDouble, outputCloned, examples)

    }
  }
  def lossAndGradients(
      samples: I,
      target: STen
  ): (Double, Long, Seq[Option[STen]]) =
    Scope.leak { implicit scope =>
      val output = module.forward(samples)
      val (loss, numInstances) = lossFunction(output, target)
      val lossAsDouble = loss.value.toMat.raw(0)

      val gradients = module.gradients(loss)
      (lossAsDouble, numInstances, gradients)
    }

  def zipOptimizer(optimizerFactory: Seq[(STen, PTag)] => Optimizer) =
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
  }
}
