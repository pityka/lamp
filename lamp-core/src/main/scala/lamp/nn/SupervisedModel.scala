package lamp.nn
import lamp.autograd.{Variable}
import lamp.Scope
import lamp.STen
import lamp.autograd.ConstantWithGrad
import lamp.Movable
import lamp.autograd.Autograd.BackpropForwardCache
import lamp.autograd.Autograd.CacheStrategy
import lamp.autograd.Autograd

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
      switchStream: Boolean,
      partialDerivatives: Map[Variable, STen],
      printMemoryAllocations: Boolean,
      cacheStrategy: CacheStrategy,
      printZeroGradients: Boolean
  )(implicit scope: Scope): (STen, Long, Map[Variable, STen])
}

class SimpleLossCalculation[I] extends LossCalculation[I] {

  def apply[M <: GenericModule[I, Variable]](
      samples: I,
      target: STen,
      module: M with GenericModule[I, Variable],
      lossFunction: LossFunction,
      computeGradients: Boolean,
      zeroGradBeforeComputingGradients: Boolean,
      switchStream: Boolean,
      partialDerivatives: Map[Variable, STen],
      printMemoryAllocations: Boolean,
      cacheStrategy: CacheStrategy,
      printZeroGradients: Boolean
  )(implicit scope: Scope): (STen, Long, Map[Variable, STen]) = {
    def body() = {
      implicit val forwardCache: BackpropForwardCache =
        if (computeGradients)
          BackpropForwardCache.empty(cacheStrategy, Nil)
        else BackpropForwardCache.empty(cacheStrategy, Nil)
      val output = module.forward(samples).persist
      val loss = lossFunction(output, target)
      val lossValue = loss.forward.cloneTensor
      val outputValue = output.forward.cloneTensor
      
      val numInstances = if (outputValue.shape.nonEmpty) outputValue.shape(0) else if (target.shape.nonEmpty) target.shape(0) else 1

        if (printMemoryAllocations) {
          println("Allocation report before backward pass.")
        println(Autograd.graphMemoryAllocationReport(loss, forwardCache,partialDerivatives))
      }

      if (computeGradients) {
        val updatedPd = module.computeGradientsAndDestroyGraph(
          loss = loss,
          pd = partialDerivatives,
          scope = scope,
          zeroGrad = zeroGradBeforeComputingGradients,
          fw = forwardCache.withCacheStrategy(CacheStrategy.AlwaysCache),
          printZeroGradients = printZeroGradients
        )

        if (printMemoryAllocations) {
          println("Allocation report after backward pass.")
        println(Autograd.graphMemoryAllocationReport(loss, forwardCache,updatedPd))
      }

        (lossValue, numInstances, updatedPd)
      } else {

        

        (lossValue, numInstances, partialDerivatives)
      }

    }
    if (switchStream)
      target.device.withOtherStream(true, true) {
        body()
      }
    else body()
  }

}

case class ParameterGradients(map: Map[ConstantWithGrad, STen]) {
  def mapParameters(s: Seq[ConstantWithGrad]) = s.map { s =>
    map.get(s) match {
      case None =>
        ???
      case Some(value) => value
    }
  }
}
object ParameterGradients {
  val empty = ParameterGradients(Map.empty)
  implicit val movable: Movable[ParameterGradients] = Movable.nonEmpty(
    _.map.flatMap(v => List(v._2.value, v._1.constantValue.value)).toList
  )
}
case class SupervisedModel[I, M <: GenericModule[I, Variable]](
    module: M with GenericModule[I, Variable],
    lossFunction: LossFunction,
    lossCalculation: LossCalculation[I] = new SimpleLossCalculation[I],
    printMemoryAllocations: Boolean = false,
    cacheStrategy : CacheStrategy = CacheStrategy.AlwaysCache,
    printZeroGradients: Boolean = false
)(implicit tm: TrainingMode[M]) {
  def asEval = copy(module = module.asEval)
  def asTraining = copy(module = module.asTraining)

  def addTotalLossAndReturnNumExamples(
      samples: I,
      target: STen,
      acc: STen,
      switchStream: Boolean
  ): Long = {

    Scope.root { implicit scope =>
      val (loss, examples, pd2) =
        lossCalculation(
          samples = samples,
          target = target,
          module = module,
          lossFunction = lossFunction,
          computeGradients = false,
          zeroGradBeforeComputingGradients = false,
          switchStream = switchStream,
          partialDerivatives = Map.empty,
          printMemoryAllocations = printMemoryAllocations,
          cacheStrategy = cacheStrategy,
          printZeroGradients = printZeroGradients
        )

      acc += (loss * examples.toDouble)
      (
        examples,
      )
    }
  }

  def addTotalLossAndReturnGradientsAndNumExamples(
      samples: I,
      target: STen,
      acc: STen,
      zeroGrad: Boolean,
      switchStream: Boolean,
      pd: ParameterGradients
  )(implicit scope: Scope): (Long, ParameterGradients) =
    Scope { implicit scope =>
      val (loss, numInstances, pd2) =
        lossCalculation(
          samples = samples,
          target = target,
          module = module,
          lossFunction = lossFunction,
          computeGradients = true,
          zeroGradBeforeComputingGradients = zeroGrad,
          switchStream = switchStream,
          partialDerivatives = pd.map.map { case (k, v) => (k: Variable, v) },
          printMemoryAllocations = printMemoryAllocations,
          cacheStrategy = cacheStrategy,
          printZeroGradients = printZeroGradients
        )
      acc += (loss * numInstances.toFloat)

      val pg2 = ParameterGradients(module.parameters.map { case (v, _) =>
        v -> pd2.get(v).getOrElse(v.allocatePartialDerivative)
      }.toMap)

      (numInstances, pg2)
    }

  def zipOptimizer(optimizerFactory: Seq[(STen, PTag)] => Optimizer) =
    ModelWithOptimizer(
      this,
      optimizerFactory(module.parameters.map(v => (v._1.constantValue, v._2)))
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
