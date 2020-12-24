package lamp.nn
import lamp.autograd.{Variable, const}
import lamp.Scope
import lamp.STen

trait LossCalculation[I] {
  def apply[M <: GenericModule[I, Variable]](
      samples: I,
      target: STen,
      module: M with GenericModule[I, Variable],
      lossFunction: LossFunction,
      computeGradients: Boolean
  )(implicit scope: Scope): (Variable, Long, Option[Seq[Option[STen]]])
}

class SimpleLossCalculation[I] extends LossCalculation[I] {

  def apply[M <: GenericModule[I, Variable]](
      samples: I,
      target: STen,
      module: M with GenericModule[I, Variable],
      lossFunction: LossFunction,
      computeGradients: Boolean
  )(implicit scope: Scope): (Variable, Long, Option[Seq[Option[STen]]]) = {
    val output = module.forward(samples)
    val (loss, numInstances) = lossFunction(output, target)

    val gradients = if (computeGradients) Some(module.gradients(loss)) else None
    (loss, numInstances, gradients)
  }

}

case class AdversarialTraining(eps: Double) extends LossCalculation[Variable] {

  def apply[M <: Module](
      samples: Variable,
      target: STen,
      module: M with Module,
      lossFunction: LossFunction,
      computeGradients: Boolean
  )(implicit scope: Scope): (Variable, Long, Option[Seq[Option[STen]]]) = {
    val samplesWithGrad = samples.withGrad
    val output0 = module.forward(samplesWithGrad)
    val (loss0, numInstances) = lossFunction(output0, target)

    val _ = module.gradients(loss0)

    val sampleGradient = samplesWithGrad.partialDerivative.get

    val adversarialSample = const(samples.value.add(sampleGradient.sign, eps))

    val adversarialOutput = module.forward(adversarialSample)
    val (adversarialLoss, _) = lossFunction(adversarialOutput, target)

    val totalLoss = (loss0 + adversarialLoss) * 0.5

    val gradients =
      if (computeGradients) Some(module.gradients(totalLoss)) else None
    (totalLoss, numInstances, gradients)
  }

}

case class SupervisedModel[I, M <: GenericModule[I, Variable]](
    module: M with GenericModule[I, Variable],
    lossFunction: LossFunction,
    lossCalculation: LossCalculation[I] = new SimpleLossCalculation[I]
)(implicit tm: TrainingMode[M]) {
  def asEval = copy(module = module.asEval)
  def asTraining = copy(module = module.asTraining)
  def addTotalLossAndReturnNumExamples(
      samples: I,
      target: STen,
      acc: STen
  ): Long = {

    Scope.leak { implicit scope =>
      val (loss, examples, _) =
        lossCalculation(samples, target, module, lossFunction, false)
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
      val (loss, numInstances, mayGradients) =
        lossCalculation(samples, target, module, lossFunction, true)
      acc += (loss.value * numInstances)

      (numInstances, mayGradients.get)
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
