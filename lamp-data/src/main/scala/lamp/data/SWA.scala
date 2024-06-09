package lamp.data

import lamp.nn._
import cats.effect._
import aten.Tensor
import scribe.Logger
import lamp.autograd.Variable
import lamp.STen
import lamp.Scope
import aten.ATen

// https://arxiv.org/pdf/1803.05407.pdf
object SWA {

  trait SWALearningRateSchedule[State] {
    def init: State
    def swaLearningRateSchedule(
        state: State,
        epoch: Int,
        lastValidationLoss: Option[Double]
    ): (State, Double, Boolean)
  }

  object SWALearningRateSchedule {

    def constant(f: Double) = new SWALearningRateSchedule[Unit] {
      def init = ()
      def swaLearningRateSchedule(
          state: Unit,
          epoch: Int,
          lastValidationLoss: Option[Double]
      ): (Unit, Double, Boolean) = ((), f, true)
    }
    def cyclic(minFactor: Double, maxFactor: Double, cycleLength: Int) =
      new SWALearningRateSchedule[Unit] {
        def init = ()
        def swaLearningRateSchedule(
            state: Unit,
            epoch: Int,
            lastValidationLoss: Option[Double]
        ): (Unit, Double, Boolean) = {
          val t = 1d / cycleLength * ((epoch % cycleLength) + 1)
          val f = (1 - t) * maxFactor + t * minFactor
          val last = if (epoch % cycleLength + 1 == cycleLength) true else false
          ((), f, last)
        }
      }
  }

  def epochs[I, M <: GenericModule[
    I,
    Variable
  ]: Load, LRState, BatchStreamState, BatchStreamBuffers](
      model: SupervisedModel[I, M],
      optimizerFactory: Seq[(STen, PTag)] => Optimizer,
      trainBatchesOverEpoch: IOLoops.TrainingLoopContext => BatchStream[
        (I, STen),
        BatchStreamState,
        BatchStreamBuffers
      ],
      validationBatchesOverEpoch: Option[
        IOLoops.TrainingLoopContext => BatchStream[
          (I, STen),
          BatchStreamState,
          BatchStreamBuffers
        ]
      ],
      epochs: Int,
      trainingCallback: Option[TrainingCallback[M]] = None,
      validationCallback: Option[ValidationCallback[M]] = None,
      checkpointState: Option[(SWALoopState, LRState) => IO[Unit]] = None,
      // checkpointLrState: Option[LRState => IO[Unit]] = None,
      validationFrequency: Int = 1,
      logger: Option[Logger] = None,
      learningRateSchedule: SWALearningRateSchedule[LRState] =
        SWALearningRateSchedule.constant(1d),
      prefetch: Boolean = false,
      dataParallelModels: Seq[SupervisedModel[I, M]] = Nil,
      initState: Option[SWALoopState] = None,
      accumulateGradientOverNBatches: Int = 1,
      learningRateScheduleInitState: Option[LRState] = None,
      forwardPassAfterTraining: Boolean = true
  ): IO[(SupervisedModel[I, M], List[(Int, Double, Option[Double])])] = {
    val modelWithOptimizer = model.asTraining.zipOptimizer(optimizerFactory)

    initState.foreach { state =>
      modelWithOptimizer.model.module.load(state.model)
      modelWithOptimizer.optimizer.load(state.optimizer)
    }

    def loop(
        epoch: Int,
        lastValidationLoss: Option[Double],
        minValidationLoss: Option[Double],
        numberOfAveragedModels: Int,
        averagedModels: Option[Seq[Tensor]],
        learningCurve: List[(Int, Double, Option[Double])],
        lrState: LRState
    ): IO[
      (
          SupervisedModel[I, M],
          List[(Int, Double, Option[Double])]
      )
    ] = {
      val (nextLRState, learningRateFactor, accumulate) =
        learningRateSchedule.swaLearningRateSchedule(
          state = lrState,
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
                  val t = ATen.div_2(tmp2, numberOfAveragedModels + 1)
                  tmp2.release
                  t
              }
          }

        }

        for {
          _ <-
            if (checkpointState.isDefined)
              checkpointState.get(
                SWALoopState(
                  modelWithOptimizer.model.module.state.map(_._1.value),
                  modelWithOptimizer.optimizer.state,
                  epoch,
                  lastValidationLoss,
                  minValidationLoss,
                  numberOfAveragedModels,
                  averagedModels,
                  learningCurve
                ),
                lrState
              )
            else IO.unit

          trainingLoss <-
            if (dataParallelModels.isEmpty)
              IOLoops.oneEpoch(
                epochCount = epoch,
                trainingCallback = trainingCallback,
                model = modelWithOptimizer,
                trainBatches = trainBatchesOverEpoch(
                  IOLoops.TrainingLoopContext(
                    epoch,
                    lastValidationLoss,
                    minValidationLoss
                  )
                ),
                logger = logger,
                learningRateScheduleFactor = learningRateFactor,
                prefetch = prefetch,
                overlapModelWithLoad = false,
                accumulateGradientOverNBatches = accumulateGradientOverNBatches
              )
            else
              DataParallel.oneEpoch(
                epoch,
                trainingCallback,
                modelWithOptimizer,
                trainBatchesOverEpoch(
                  IOLoops.TrainingLoopContext(
                    epoch,
                    lastValidationLoss,
                    minValidationLoss
                  )
                ),
                logger,
                learningRateFactor,
                dataParallelModels,
                accumulateGradientOverNBatches
              )

          maybeValidationLoss <-
            if (
              epoch % validationFrequency == 0 && validationBatchesOverEpoch.isDefined
            )
              if (dataParallelModels.isEmpty)
                IOLoops
                  .validationOneEpoch(
                    model = modelWithOptimizer.model,
                    validationBatches = validationBatchesOverEpoch.get(
                      IOLoops.TrainingLoopContext(
                        epoch,
                        lastValidationLoss,
                        minValidationLoss
                      )
                    ),
                    validationCallback = validationCallback,
                    logger = logger,
                    epochCount = epoch
                  )
                  .map(Some(_))
              else
                DataParallel
                  .validationOneEpoch(
                    models = modelWithOptimizer.model +: dataParallelModels,
                    validationBatches = validationBatchesOverEpoch.get(
                      IOLoops.TrainingLoopContext(
                        epoch,
                        lastValidationLoss,
                        minValidationLoss
                      )
                    ),
                    validationCallback = validationCallback,
                    logger = logger,
                    epochCount = epoch
                  )
                  .map(Some(_))
            else IO.pure(None)

          _ <- IO {
            maybeValidationLoss.foreach(validationLoss =>
              validationCallback.foreach(_.apply(epoch, validationLoss, model.module))
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
              (epoch, trainingLoss, maybeValidationLoss) :: learningCurve,
            lrState = nextLRState
          )
        } yield next
      }
    }

    for {
      trained <- initState match {
        case None =>
          loop(
            0,
            None,
            None,
            0,
            None,
            Nil,
            learningRateScheduleInitState.getOrElse(learningRateSchedule.init)
          )
        case Some(state) =>
          loop(
            epoch = state.epoch,
            lastValidationLoss = state.lastValidationLoss,
            minValidationLoss = state.minValidationLoss,
            numberOfAveragedModels = state.numberOfAveragedModels,
            averagedModels = state.averagedModels,
            learningCurve = state.learningCurve,
            lrState =
              learningRateScheduleInitState.getOrElse(learningRateSchedule.init)
          )
      }

      (model, learningCurve) = trained
      // update batchnorm's state in a side effect
      _ <-
        if (forwardPassAfterTraining) {
          val batchStream = trainBatchesOverEpoch(
            IOLoops.TrainingLoopContext(
              learningCurve.size,
              None,
              None
            )
          )
          IOLoops
            .forwardAndDiscardBatchStream(
              batchStream,
              batchStream.allocateBuffers,
              model.module
            )
        } else IO.unit
    } yield trained

  }
}
