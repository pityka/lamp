package lamp.data
import lamp.nn._
import cats.effect._
import aten.Tensor
import scribe.Logger
import lamp.autograd.Variable
import lamp.STen
import lamp.Scope
import cats.effect.std.Queue

/** Contains a training loops and helpers around it
  *
  * The two training loops implemented here are:
  *   - [[lamp.data.IOLoops.epochs]]
  *   - [[lamp.data.IOLoops.withSWA]] implements Stochastic Weight Averaging
  */
object IOLoops {

  def forwardBatchStream[I, M <: GenericModule[I, Variable]](
      batchStream: BatchStream[I],
      model: M with GenericModule[I, Variable]
  ): IO[Unit] = {

    val device = model.state.head._1.value.device

    def loop(
        batch: Resource[IO, StreamControl[(I, STen)]]
    ): IO[Unit] = {
      batch.use {
        case EndStream  => IO.pure(false)
        case EmptyBatch => IO.pure(true)
        case NonEmptyBatch((x, _)) =>
          IO { Scope.root { implicit scope => model.forward(x) }; true }
      } flatMap {
        case true  => loop(batchStream.nextBatch(device))
        case false => IO.unit
      }
    }

    loop(batchStream.nextBatch(device))
  }
  def runBatchStream[I, M <: GenericModule[I, Variable]](
      batchStream: BatchStream[I],
      model: M with GenericModule[I, Variable]
  )(implicit scope: Scope) = {

    val device = model.state.head._1.value.device

    def loop(
        batch: Resource[IO, StreamControl[(I, STen)]],
        acc: List[STen]
    ): IO[List[STen]] = {
      batch.use {
        case EndStream  => IO.pure(Left(acc.reverse))
        case EmptyBatch => IO.pure(Right(acc))
        case NonEmptyBatch((x, _)) =>
          IO {
            Right(Scope { implicit scope =>
              model.forward(x).value
            } :: acc)
          }

      } flatMap {
        case Left(acc)  => IO.pure(acc)
        case Right(acc) => loop(batchStream.nextBatch(device), acc)
      }

    }

    loop(batchStream.nextBatch(device), Nil)
  }

  def withSWA[I, M <: GenericModule[I, Variable]: Load, LRState, LRStateSWA](
      model: SupervisedModel[I, M],
      optimizerFactory: Seq[(STen, PTag)] => Optimizer,
      trainBatchesOverEpoch: () => BatchStream[I],
      warmupEpochs: Int,
      swaEpochs: Int,
      validationBatchesOverEpoch: Option[() => BatchStream[I]] = None,
      trainingCallback: TrainingCallback = TrainingCallback.noop,
      validationCallback: ValidationCallback = ValidationCallback.noop,
      checkpointState: Option[LoopState => IO[Unit]] = None,
      checkpointLRState: Option[LRState => IO[Unit]] = None,
      checkpointSWALRState: Option[LRStateSWA => IO[Unit]] = None,
      logger: Option[Logger] = None,
      returnMinValidationLossModel: Seq[Int] = Nil,
      learningRateSchedule: LearningRateSchedule[LRState] =
        LearningRateSchedule.decrement(20, 0.5),
      swaLearningRateSchedule: SWA.SWALearningRateSchedule[LRStateSWA] =
        SWA.SWALearningRateSchedule.cyclic(
          minFactor = 0.01,
          maxFactor = 1d,
          cycleLength = 10
        ),
      prefetch: Boolean = false,
      dataParallelModels: Seq[SupervisedModel[I, M]] = Nil,
      initState: Option[SimpleThenSWALoopState] = None,
      accumulateGradientOverNBatches: Int = 1,
      learningRateScheduleInitState: Option[LRState] = None,
      swaLearningRateScheduleInitState: Option[LRStateSWA] = None
  ) = {
    for {
      warmedup <-
        epochs(
          model,
          optimizerFactory,
          trainBatchesOverEpoch,
          validationBatchesOverEpoch,
          warmupEpochs,
          trainingCallback,
          validationCallback,
          checkpointState,
          checkpointLRState,
          1,
          logger,
          returnMinValidationLossModel,
          learningRateSchedule,
          prefetch,
          dataParallelModels,
          initState.flatMap(_.simple),
          accumulateGradientOverNBatches,
          learningRateScheduleInitState
        )

      warmupEpochReturned = warmedup._1
      warmedupModel = warmedup._2
      warmupLearningCurve = warmedup._3
      swaResult <- SWA.epochs(
        warmedupModel,
        optimizerFactory,
        trainBatchesOverEpoch,
        validationBatchesOverEpoch,
        swaEpochs,
        trainingCallback,
        validationCallback,
        checkpointState,
        checkpointSWALRState,
        1,
        logger,
        swaLearningRateSchedule,
        prefetch,
        dataParallelModels,
        initState.flatMap(_.swa),
        accumulateGradientOverNBatches,
        swaLearningRateScheduleInitState
      )
    } yield {
      val swaModel = swaResult._1
      val swaLearningCurve = swaResult._2
      val m = warmupLearningCurve.map(_._1).max + 1
      val concat = warmupLearningCurve ++ swaLearningCurve.map {
        case (epoch, l1, l2) => (epoch + m, l1, l2)
      }
      (warmupEpochReturned, swaModel, concat)
    }
  }

  def epochs[I, M <: GenericModule[I, Variable]: Load, LRState](
      model: SupervisedModel[I, M],
      optimizerFactory: Seq[(STen, PTag)] => Optimizer,
      trainBatchesOverEpoch: () => BatchStream[I],
      validationBatchesOverEpoch: Option[() => BatchStream[I]],
      epochs: Int,
      trainingCallback: TrainingCallback = TrainingCallback.noop,
      validationCallback: ValidationCallback = ValidationCallback.noop,
      checkpointState: Option[LoopState => IO[Unit]] = None,
      checkpointLRState: Option[LRState => IO[Unit]] = None,
      validationFrequency: Int = 1,
      logger: Option[Logger] = None,
      returnMinValidationLossModel: Seq[Int] = Nil,
      learningRateSchedule: LearningRateSchedule[LRState] =
        LearningRateSchedule.noop,
      prefetch: Boolean = false,
      dataParallelModels: Seq[SupervisedModel[I, M]] = Nil,
      initState: Option[SimpleLoopState] = None,
      accumulateGradientOverNBatches: Int = 1,
      learningRateScheduleInitState: Option[LRState] = None
  ): IO[(Int, SupervisedModel[I, M], List[(Int, Double, Option[Double])])] = {
    val modelWithOptimizer
        : ModelWithOptimizer[I, M with GenericModule[I, Variable]] =
      model.asTraining.zipOptimizer(optimizerFactory)

    initState.foreach { case state =>
      modelWithOptimizer.model.module.load(state.model)
      modelWithOptimizer.optimizer.load(state.optimizer)
    }

    def loop(
        epoch: Int,
        lastValidationLoss: Option[Double],
        minValidationLoss: Option[Double],
        minValidationLossModel: Option[(Int, Seq[Tensor])],
        learningCurve: List[(Int, Double, Option[Double])],
        lrState: LRState
    ): IO[
      (Int, SupervisedModel[I, M], List[(Int, Double, Option[Double])])
    ] = {
      val (nextLearningRateScheduleState, learningRateFactor) =
        learningRateSchedule.learningRateFactor(
          state = lrState,
          epoch = epoch,
          lastValidationLoss = lastValidationLoss
        )
      if (epoch >= epochs || learningRateFactor <= 0d)
        IO.pure {
          modelWithOptimizer.optimizer.release()
          minValidationLossModel match {
            case None =>
              (epoch - 1, modelWithOptimizer.model, learningCurve.reverse)
            case Some((epochOfMinValidation, state)) =>
              Scope.root { implicit scope =>
                val stateOnDevice = state.map { t => STen.owned(t) }
                model.module.load(stateOnDevice)
              }
              (
                epochOfMinValidation,
                modelWithOptimizer.model,
                learningCurve.reverse
              )
          }
        }
      else {

        def copyModel = {

          logger.foreach(_.info(s"Copying model at epoch $epoch"))
          minValidationLossModel.foreach(_._2.foreach(_.release))
          val copiedState =
            model.module.state.map(_._1.value).map { t =>
              aten.ATen.clone(t.value)
            }

          (epoch, copiedState)
        }

        for {
          _ <-
            if (checkpointState.isDefined)
              checkpointState.get(
                SimpleLoopState(
                  modelWithOptimizer.model.module.state.map(_._1.value),
                  modelWithOptimizer.optimizer.state,
                  epoch,
                  lastValidationLoss,
                  minValidationLoss,
                  minValidationLossModel,
                  learningCurve
                )
              )
            else IO.unit
          _ <-
            if (checkpointLRState.isDefined)
              checkpointLRState.get(
                lrState
              )
            else IO.unit

          trainingLoss <-
            if (dataParallelModels.isEmpty)
              oneEpoch(
                epoch,
                trainingCallback,
                modelWithOptimizer,
                trainBatchesOverEpoch(),
                logger,
                learningRateFactor,
                prefetch,
                accumulateGradientOverNBatches
              )
            else
              DataParallel.oneEpoch(
                epoch,
                trainingCallback,
                modelWithOptimizer,
                trainBatchesOverEpoch(),
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
                validationOneEpoch(
                  model = modelWithOptimizer.model,
                  validationBatches = validationBatchesOverEpoch.get(),
                  validationCallback = validationCallback,
                  logger = logger,
                  epochCount = epoch
                ).map(Some(_))
              else
                DataParallel
                  .validationOneEpoch(
                    models = modelWithOptimizer.model +: dataParallelModels,
                    validationBatches = validationBatchesOverEpoch.get(),
                    validationCallback = validationCallback,
                    logger = logger,
                    epochCount = epoch
                  )
                  .map(Some(_))
            else IO.pure(None)

          _ <- IO {
            maybeValidationLoss.foreach(validationLoss =>
              validationCallback.apply(epoch, validationLoss)
            )
          }

          nextMinValidationLoss =
            if (
              maybeValidationLoss.isEmpty || !returnMinValidationLossModel
                .contains(epoch)
            )
              minValidationLoss
            else if (minValidationLoss.isEmpty) maybeValidationLoss
            else
              Some(math.min(minValidationLoss.get, maybeValidationLoss.get))

          nextMinValidationLossModel =
            if (
              returnMinValidationLossModel
                .contains(epoch)
            ) {
              if (maybeValidationLoss.isEmpty) minValidationLossModel
              else if (minValidationLoss.isEmpty) Some(copyModel)
              else if (minValidationLoss.get > maybeValidationLoss.get)
                Some(copyModel)
              else minValidationLossModel
            } else minValidationLossModel
          next <- loop(
            epoch = epoch + 1,
            lastValidationLoss = maybeValidationLoss,
            minValidationLoss = nextMinValidationLoss,
            minValidationLossModel = nextMinValidationLossModel,
            learningCurve =
              (epoch, trainingLoss, maybeValidationLoss) :: learningCurve,
            lrState = nextLearningRateScheduleState
          )
        } yield next
      }
    }

    initState match {
      case None =>
        loop(0, None, None, None, Nil, learningRateSchedule.init)
      case Some(state) =>
        loop(
          epoch = state.epoch,
          lastValidationLoss = state.lastValidationLoss,
          minValidationLoss = state.minValidationLoss,
          minValidationLossModel = state.minValidationLossModel,
          learningCurve = state.learningCurve,
          lrState =
            learningRateScheduleInitState.getOrElse(learningRateSchedule.init)
        )
    }

  }

  def oneEpoch[I, M <: GenericModule[I, Variable]](
      epochCount: Long,
      trainingCallback: TrainingCallback,
      model: ModelWithOptimizer[I, M],
      trainBatches: BatchStream[I],
      logger: Option[Logger],
      learningRateScheduleFactor: Double,
      prefetch: Boolean,
      accumulateGradientOverNBatches: Int
  ): IO[Double] = {

    val device = model.model.module.state.head._1.value.device

    def processBatch(
        elem: StreamControl[(I, STen)],
        lossAcc: STen,
        batchCount: Long
    ): StreamControl[Long] = elem.map { case (sample, target) =>
      if (accumulateGradientOverNBatches <= 1) {
        val (numInstances, gradients) =
          model.model.addTotalLossAndReturnGradientsAndNumExamples(
            sample,
            target,
            lossAcc,
            zeroGrad = true
          )

        model.optimizer.step(gradients, learningRateScheduleFactor)
        numInstances
      } else {

        val (numInstances, gradients) =
          model.model.addTotalLossAndReturnGradientsAndNumExamples(
            sample,
            target,
            lossAcc,
            zeroGrad = false
          )

        if (
          batchCount > 0 && batchCount % accumulateGradientOverNBatches == 0
        ) {
          model.optimizer.step(gradients, learningRateScheduleFactor)
          model.model.zeroGrad()
        }

        numInstances

      }

    }

    def simpleLoop(
        lossAcc: STen,
        numInstancesAcc: Long,
        batchCount: Long
    ): IO[Long] = {
      trainBatches
        .nextBatch(device)
        .use { batch => IO { processBatch(batch, lossAcc, batchCount) } }
        .flatMap {
          case EndStream  => IO.pure(numInstancesAcc)
          case EmptyBatch => simpleLoop(lossAcc, numInstancesAcc, batchCount)
          case NonEmptyBatch(numInstances) =>
            simpleLoop(
              lossAcc,
              numInstances + numInstancesAcc,
              batchCount + 1L
            )
        }
    }

    def prefetch1[A, B](
        fetch: () => Resource[IO, StreamControl[A]],
        transform: (Long, StreamControl[A]) => IO[StreamControl[B]],
        reduce: (B, B) => B,
        zero: B
    ): IO[B] = {

      def loop(
          counter: Long,
          acc: B,
          queue: Queue[IO, (StreamControl[A], IO[Unit])]
      ): IO[B] = {
        for {
          fetched <- queue.take
          a = fetched._1
          release = fetched._2
          _ <- IO { fetch() }
            .flatMap(resource => resource.allocated.flatMap(queue.offer))
            .start
          done <- transform(counter, a)
          _ <- release
          loopDone <- done match {
            case EndStream  => IO.pure(acc)
            case EmptyBatch => loop(counter, acc, queue)
            case NonEmptyBatch(b) =>
              loop(counter + 1, reduce(b, acc), queue)
          }
        } yield loopDone
      }

      for {
        q <- Queue.bounded[IO, (StreamControl[A], IO[Unit])](1)
        _ <- IO { fetch() }.flatMap(_.allocated.flatMap(q.offer)).start
        l <- loop(0, zero, q)
      } yield l

    }

    def prefetchLoop(
        lossAcc: STen
    ) = {

      prefetch1[(I, STen), Long](
        fetch = () => trainBatches.nextBatch(device),
        transform = (batchCounter, batch) =>
          IO {
            processBatch(batch, lossAcc, batchCounter)
          },
        reduce = (b, acc) => (acc + b),
        zero = 0L
      )
    }

    val epochLoop = Scope.inResource.use { implicit scope =>
      val lossAcc =
        STen.scalarDouble(0d, model.model.module.state.head._1.options)
      val loopDone =
        if (prefetch)
          prefetchLoop(lossAcc)
        else simpleLoop(lossAcc, 0L, 0L)

      loopDone.map { numInstances =>
        val totalLoss = lossAcc.toMat.raw(0)
        (totalLoss, numInstances)
      }
    }
    for {
      pair <- epochLoop
      (totalLoss, numInstances) = pair
      trainingLoss = totalLoss / numInstances

      _ <- IO {
        logger.foreach(
          _.info(
            s"Avg training loss in epoch $epochCount over $numInstances examples: $trainingLoss"
          )
        )
      }
      _ <- IO {
        trainingCallback(epochCount, trainingLoss)
      }

    } yield trainingLoss

  }
  def validationOneEpoch[I, M <: GenericModule[I, Variable]](
      model: SupervisedModel[I, M],
      validationBatches: BatchStream[I],
      validationCallback: ValidationCallback,
      logger: Option[Logger],
      epochCount: Long
  ): IO[Double] = {
    val device = model.module.state.head._1.value.device
    val modelAsEval = model.asEval

    def loop(
        batchCount: Int,
        totalLoss: STen,
        totalExamples: Long
    ): IO[(STen, Long)] = {
      validationBatches
        .nextBatch(device)
        .use { elem =>
          IO {
            elem.map { case (validationSample, validationTarget) =>
              val numExamples =
                modelAsEval.addTotalLossAndReturnNumExamples(
                  validationSample,
                  validationTarget,
                  totalLoss
                )
              numExamples
            }
          }
        }
        .flatMap {
          case EndStream  => IO.pure((totalLoss, totalExamples))
          case EmptyBatch => loop(batchCount, totalLoss, totalExamples)
          case NonEmptyBatch(examples) =>
            loop(
              batchCount + 1,
              totalLoss,
              totalExamples + examples
            )
        }

    }

    Scope.inResource.use { implicit scope =>
      loop(
        0,
        STen.scalarDouble(0d, model.module.state.head._1.options),
        0L
      ).flatMap { case (totalLoss, totalExamples) =>
        val validationLoss = totalLoss.toMat.raw(0) / totalExamples
        for {
          _ <- IO {
            logger.foreach(
              _.info(
                s"Avg validation loss in epoch $epochCount over $totalExamples examples: ${validationLoss}"
              )
            )
          }
          _ <- IO {
            validationCallback(epochCount, validationLoss)
          }

        } yield validationLoss
      }
    }
  }

}
