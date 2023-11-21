package lamp.nn
import lamp.autograd.{Variable, const}
import lamp.Scope
import lamp.STen

/** Loss and Gradient calculation
  *
  * Takes samples, target, module, loss function and computes the loss and the
  * gradients
  */
trait LossCalculation[I] {
  def apply[M <: GenericModule[I, Variable]](
      samples: I,
      target: STen,
      module: M with GenericModule[I, Variable],
      lossFunction: LossFunction,
      computeGradients: Boolean,
      zeroGradBeforeComputingGradients: Boolean,
      switchStream: Boolean
  )(implicit scope: Scope): (Variable, Long, Option[Seq[Option[STen]]])
}

/** Evaluates the gradient at current point + eps where eps is I *
  * N(0,noiseLevel)
  */
class PerturbedLossCalculation[I](noiseLevel: Double)
    extends LossCalculation[I] {

  private def perturb(noiseLevel: Double, params: Seq[STen])(implicit
      scope: Scope
  ) = {

    params.foreach { case param =>
      val n = STen.randn(param.shape, param.options)
      n *= noiseLevel
      param += n
    }
  }

  private def saveState[T](
      params: Seq[STen]
  )(f: => T)(implicit scope: Scope): T = {
    val copy = params.map(_.cloneTensor)
    val r = f
    params.zip(copy).foreach { case (orig, copy) =>
      orig.copyFrom(copy)
    }
    r
  }

  def apply[M <: GenericModule[I, Variable]](
      samples: I,
      target: STen,
      module: M with GenericModule[I, Variable],
      lossFunction: LossFunction,
      computeGradients: Boolean,
      zeroGradBeforeComputingGradients: Boolean,
      switchStream: Boolean
  )(implicit scope: Scope): (Variable, Long, Option[Seq[Option[STen]]]) = {
    val gradients = Scope { implicit scope =>
      saveState(module.parameters.map(_._1.value)) {
        perturb(
          noiseLevel,
          module.parameters.map(_._1.value)
        )
        Scope { implicit scope =>
          val (perturbedLoss, _) = lossFunction(module.forward(samples), target)

          if (computeGradients)
            Some(
              module.gradients(perturbedLoss, zeroGradBeforeComputingGradients)
            )
          else None
        }

      }
    }
    val (realLoss, numInstances) =
      lossFunction(module.forward(samples), target)
    (realLoss, numInstances, gradients)
  }

}
class SimpleLossCalculation[I] extends LossCalculation[I] {

  def apply[M <: GenericModule[I, Variable]](
      samples: I,
      target: STen,
      module: M with GenericModule[I, Variable],
      lossFunction: LossFunction,
      computeGradients: Boolean,
      zeroGradBeforeComputingGradients: Boolean,
      switchStream: Boolean
  )(implicit scope: Scope): (Variable, Long, Option[Seq[Option[STen]]]) = {
    def body() = {

      val output = module.forward(samples)
      val (loss, numInstances) = lossFunction(output, target)

      val gradients =
        if (computeGradients)
          Some(module.gradients(loss, zeroGradBeforeComputingGradients))
        else None

      (loss, numInstances, gradients)
    }
    if (switchStream)
      target.device.withOtherStream(true, true) {
        body()
      }
    else body()
  }

}

case class AdversarialTraining(eps: Double) extends LossCalculation[Variable] {

  def apply[M <: Module](
      samples: Variable,
      target: STen,
      module: M with Module,
      lossFunction: LossFunction,
      computeGradients: Boolean,
      zeroGradBeforeComputingGradients: Boolean,
      switchStream: Boolean
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
      if (computeGradients)
        Some(module.gradients(totalLoss, zeroGradBeforeComputingGradients))
      else None
    (totalLoss, numInstances, gradients)
  }

}

case class SupervisedModel[I, M <: GenericModule[I, Variable]](
    module: M with GenericModule[I, Variable],
    lossFunction: LossFunction,
    lossCalculation: LossCalculation[I] = new SimpleLossCalculation[I],
    printMemoryAllocations: Boolean = false
)(implicit tm: TrainingMode[M]) {
  def asEval = copy(module = module.asEval)
  def asTraining = copy(module = module.asTraining)

  def zeroGrad() = {
    module.zeroGrad()
  }

  def addTotalLossAndReturnNumExamples(
      samples: I,
      target: STen,
      acc: STen,
      switchStream: Boolean
  ): Long = {

    Scope.root { implicit scope =>
      val (loss, examples, _) =
        lossCalculation(
          samples,
          target,
          module,
          lossFunction,
          false,
          false,
          switchStream
        )
      if (printMemoryAllocations) {
        println(loss.graphMemoryAllocationReport)
      }
      acc += (loss.value * examples.toDouble)
      examples
    }
  }

  def addTotalLossAndReturnGradientsAndNumExamples(
      samples: I,
      target: STen,
      acc: STen,
      zeroGrad: Boolean,
      switchStream: Boolean
  ): (Long, Seq[Option[STen]]) =
    Scope.unsafe { implicit scope =>
      val (loss, numInstances, mayGradients) =
        lossCalculation(
          samples,
          target,
          module,
          lossFunction,
          true,
          zeroGrad,
          switchStream
        )
      acc += (loss.value * numInstances.toDouble)

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
    optimizer.release()
  }
}
