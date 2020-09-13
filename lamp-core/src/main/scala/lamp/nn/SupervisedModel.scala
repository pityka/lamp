package lamp.nn
import aten.Tensor
import lamp.autograd.Variable
import aten.ATen
import cats.effect.Resource
import cats.effect.IO
import lamp.syntax
import lamp.autograd.Mean

trait ComputeLoss[I] {
  def computeOutputAndLoss[M <: GenericModule[I, Variable]](
      module: M with GenericModule[I, Variable],
      lossFunction: LossFunction,
      input: I,
      target: Tensor
  ): (Variable, Variable, Long)
}

class SimpleComputeLoss[I] extends ComputeLoss[I] {
  def computeOutputAndLoss[M <: GenericModule[I, Variable]](
      module: M with GenericModule[I, Variable],
      lossFunction: LossFunction,
      input: I,
      target: Tensor
  ): (Variable, Variable, Long) = {
    val output = module.forward(input)
    val (loss, examples) = lossFunction(output, target, Mean)
    (output, loss, examples)
  }
}

case class InputGradientRegularizer(h: Double, lambda: Double)
    extends ComputeLoss[Variable] {
  def computeOutputAndLoss[M <: GenericModule[Variable, Variable]](
      module: M with GenericModule[Variable, Variable],
      lossFunction: LossFunction,
      input: Variable,
      target: Tensor
  ): (Variable, Variable, Long) = {
    InputGradientRegularization.outputAndLoss(module, lossFunction)(
      input,
      target,
      h,
      lambda
    )

  }
}

object InputGradient {
  def computeInputGradient[M <: GenericModule[Variable, Variable]](
      model: SupervisedModel[Variable, M],
      input: Variable,
      target: Tensor
  ) = {
    val inputWithNeedGrad = input.copy(needsGrad = true)

    val (_, loss, _) = (new SimpleComputeLoss[Variable])
      .computeOutputAndLoss(
        model.module,
        model.lossFunction,
        inputWithNeedGrad,
        target
      )

    model.module.parameters.foreach {
      case (param, _) =>
        param.zeroGrad()
    }
    inputWithNeedGrad.zeroGrad()

    loss.backprop()
    val g = inputWithNeedGrad.partialDerivative.get
    loss.releaseAll
    g
  }
}

object SupervisedModel {
  def apply[I, M <: GenericModule[I, Variable]](
      module: M with GenericModule[I, Variable],
      lossFunction: LossFunction
  )(implicit tm: TrainingMode[M]): SupervisedModel[I, M] =
    SupervisedModel(module, lossFunction, new SimpleComputeLoss)
}

case class SupervisedModel[I, M <: GenericModule[I, Variable]](
    module: M with GenericModule[I, Variable],
    lossFunction: LossFunction,
    computeLoss: ComputeLoss[I]
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

      val (output, loss, examples) =
        computeLoss.computeOutputAndLoss(module, lossFunction, samples, target)
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

    val (output, loss, _) =
      computeLoss.computeOutputAndLoss(module, lossFunction, samples, target)
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
