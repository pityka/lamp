package lamp.nn
import lamp.autograd.Variable
import lamp.Scope
import lamp.STen

case class SupervisedModel[I, M <: GenericModule[I, Variable]](
    module: M with GenericModule[I, Variable],
    lossFunction: LossFunction
)(implicit tm: TrainingMode[M]) {
  def asEval = copy(module = module.asEval)
  def asTraining = copy(module = module.asTraining)
  def addTotalLossAndReturnNumExamples(
      samples: I,
      target: STen,
      acc: STen
  ): Long = {

    Scope.leak { implicit scope =>
      val output = module.forward(samples)
      val (loss, examples) = lossFunction(output, target)
      acc += (loss.value * examples)
      examples
    }
  }

  def addTotalLossAndReturnGradientsAndNumExamples(
      samples: I,
      target: STen,
      acc: STen
  ): (Long, Seq[Option[STen]]) =
    Scope.leak { implicit scope =>
      val output = module.forward(samples)
      val (loss, numInstances) = lossFunction(output, target)
      acc += (loss.value * numInstances)

      val gradients = module.gradients(loss)
      (numInstances, gradients)
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
