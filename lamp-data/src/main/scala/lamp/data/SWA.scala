package lamp.data

import lamp.nn._
import cats.effect._
import aten.Tensor
import java.io.File
import scribe.Logger
import Writer.writeCheckpoint
import lamp.autograd.Variable
import lamp.STen
import lamp.Scope
import aten.ATen

// https://arxiv.org/pdf/1803.05407.pdf
object SWA {

  trait SWALearningRateSchedule {
    def swaLearningRateSchedule(
        epoch: Int,
        lastValidationLoss: Option[Double]
    ): (Double, Boolean)
  }

  object SWALearningRateSchedule {
    def constant(f: Double) = new SWALearningRateSchedule {
      def swaLearningRateSchedule(
          epoch: Int,
          lastValidationLoss: Option[Double]
      ): (Double, Boolean) = (f, true)
    }
    def cyclic(minFactor: Double, maxFactor: Double, cycleLength: Int) =
      new SWALearningRateSchedule {
        def swaLearningRateSchedule(
            epoch: Int,
            lastValidationLoss: Option[Double]
        ): (Double, Boolean) = {
          val t = 1d / cycleLength * ((epoch % cycleLength) + 1)
          val f = (1 - t) * maxFactor + t * minFactor
          val last = if (epoch % cycleLength + 1 == cycleLength) true else false
          (f, last)
        }
      }
  }

  def epochs[I, M <: GenericModule[I, Variable]: Load](
      model: SupervisedModel[I, M],
      optimizerFactory: Seq[(STen, PTag)] => Optimizer,
      trainBatchesOverEpoch: () => BatchStream[I],
      validationBatchesOverEpoch: Option[() => BatchStream[I]],
      epochs: Int,
      trainingCallback: TrainingCallback = TrainingCallback.noop,
      validationCallback: ValidationCallback = ValidationCallback.noop,
      checkpointFile: Option[File] = None,
      minimumCheckpointFile: Option[File] = None,
      validationFrequency: Int = 1,
      logger: Option[Logger] = None,
      learningRateSchedule: SWALearningRateSchedule =
        SWALearningRateSchedule.constant(1d),
      prefetch: Boolean = false
  ): IO[(SupervisedModel[I, M], List[(Int, Double, Option[Double])])] = {
    val modelWithOptimizer = model.asTraining.zipOptimizer(optimizerFactory)

    def loop(
        epoch: Int,
        lastValidationLoss: Option[Double],
        minValidationLoss: Option[Double],
        numberOfAveragedModels: Int,
        averagedModels: Option[Seq[Tensor]],
        learningCurve: List[(Int, Double, Option[Double])]
    ): IO[
      (
          SupervisedModel[I, M],
          List[(Int, Double, Option[Double])]
      )
    ] = {
      val (learningRateFactor, accumulate) =
        learningRateSchedule.swaLearningRateSchedule(
          epoch = epoch,
          lastValidationLoss = lastValidationLoss
        )
      if (epoch >= epochs || learningRateFactor <= 0d)
        IO.pure {
          modelWithOptimizer.optimizer.release()
          averagedModels match {
            case None =>
              (modelWithOptimizer.model, learningCurve.reverse)
            case Some(state) =>
              Scope.root { implicit scope =>
                val stateOnDevice = state.map { t => STen.owned(t) }
                model.module.load(stateOnDevice)
              }
              (
                modelWithOptimizer.model,
                learningCurve.reverse
              )
          }
        }
      else {

        def updateAccumulator = {

          averagedModels match {
            case None =>
              model.module.state.map(st => ATen.clone(st._1.value.value))
            case Some(avg) =>
              model.module.state.map(_._1.value.value).zip(avg).map {
                case (current, avg) =>
                  val tmp1 = ATen.mul_1(avg, numberOfAveragedModels)
                  avg.release
                  val tmp2 = ATen.add_0(tmp1, current, 1d)
                  tmp1.release
                  val t = ATen.div_1(tmp2, numberOfAveragedModels + 1)
                  tmp2.release
                  t
              }
          }

        }

        for {
          trainingLoss <- IOLoops.oneEpoch(
            epoch,
            trainingCallback,
            modelWithOptimizer,
            trainBatchesOverEpoch(),
            logger,
            learningRateFactor,
            prefetch
          )

          _ <-
            if (checkpointFile.isDefined)
              writeCheckpoint(checkpointFile.get, model.module)
            else IO.unit
          maybeValidationLoss <-
            if (
              epoch % validationFrequency == 0 && validationBatchesOverEpoch.isDefined
            )
              IOLoops
                .validationOneEpoch(
                  model = modelWithOptimizer.model,
                  validationBatches = validationBatchesOverEpoch.get(),
                  validationCallback = validationCallback,
                  logger = logger,
                  epochCount = epoch,
                  minimumCheckpointFile = minimumCheckpointFile,
                  minimumValidationLossSoFar = minValidationLoss
                )
                .map(Some(_))
            else IO.pure(None)

          _ <- IO {
            maybeValidationLoss.foreach(validationLoss =>
              validationCallback.apply(epoch, validationLoss)
            )
          }

          nextMinValidationLoss =
            if (maybeValidationLoss.isEmpty)
              minValidationLoss
            else if (minValidationLoss.isEmpty) maybeValidationLoss
            else
              Some(math.min(minValidationLoss.get, maybeValidationLoss.get))

          nextAveragedModel =
            if (accumulate) Some(updateAccumulator)
            else averagedModels

          nextNumberOfAveragedModels =
            if (accumulate)
              numberOfAveragedModels + 1
            else numberOfAveragedModels

          next <- loop(
            epoch = epoch + 1,
            lastValidationLoss = maybeValidationLoss,
            minValidationLoss = nextMinValidationLoss,
            averagedModels = nextAveragedModel,
            numberOfAveragedModels = nextNumberOfAveragedModels,
            learningCurve =
              (epoch, trainingLoss, maybeValidationLoss) :: learningCurve
          )
        } yield next
      }
    }

    for {
      trained <- loop(0, None, None, 0, None, Nil)
      (model, learningCurve) = trained
      // update batchnorm's state in a side effect
      _ <- IOLoops
        .forwardBatchStream(trainBatchesOverEpoch(), model.module)
    } yield trained

  }
}
