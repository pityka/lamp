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
import scala.concurrent.ExecutionContext
import java.util.concurrent.Executors

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

    def loop(
        batch: Resource[IO, StreamControl[(I, STen)]]
    ): IO[Unit] = {
      batch.use {
        case EndStream  => IO.pure(false)
        case EmptyBatch => IO.pure(true)
        case NonEmptyBatch((x, _)) =>
          IO { Scope.root { implicit scope => model.forward(x) }; true }
      } flatMap {
        case true  => loop(batchStream.nextBatch)
        case false => IO.unit
      }
    }

    loop(batchStream.nextBatch)
  }
  def runBatchStream[I, M <: GenericModule[I, Variable]](
      batchStream: BatchStream[I],
      model: M with GenericModule[I, Variable]
  )(implicit scope: Scope) = {

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
        case Right(acc) => loop(batchStream.nextBatch, acc)
      }

    }

    loop(batchStream.nextBatch, Nil)
  }

  def withSWA[I, M <: GenericModule[I, Variable]: Load](
      model: SupervisedModel[I, M],
      optimizerFactory: Seq[(STen, PTag)] => Optimizer,
      trainBatchesOverEpoch: () => BatchStream[I],
      warmupEpochs: Int,
      swaEpochs: Int,
      validationBatchesOverEpoch: Option[() => BatchStream[I]] = None,
      trainingCallback: TrainingCallback = TrainingCallback.noop,
      validationCallback: ValidationCallback = ValidationCallback.noop,
      checkpointFile: Option[File] = None,
      minimumCheckpointFile: Option[File] = None,
      logger: Option[Logger] = None,
      returnMinValidationLossModel: Seq[Int] = Nil,
      learningRateSchedule: LearningRateSchedule =
        LearningRateSchedule.decrement(20, 0.5),
      swaLearningRateSchedule: SWA.SWALearningRateSchedule =
        SWA.SWALearningRateSchedule.cyclic(
          minFactor = 0.01,
          maxFactor = 1d,
          cycleLength = 10
        ),
      prefetchData: Boolean = true
  ) = {
    for {
      warmedup <- epochs(
        model,
        optimizerFactory,
        trainBatchesOverEpoch,
        validationBatchesOverEpoch,
        warmupEpochs,
        trainingCallback,
        validationCallback,
        checkpointFile,
        minimumCheckpointFile,
        1,
        logger,
        returnMinValidationLossModel,
        learningRateSchedule,
        prefetchData
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
        checkpointFile,
        minimumCheckpointFile,
        1,
        logger,
        swaLearningRateSchedule,
        prefetchData
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
      returnMinValidationLossModel: Seq[Int] = Nil,
      learningRateSchedule: LearningRateSchedule = LearningRateSchedule.noop,
      prefetchData: Boolean = false
  ): IO[(Int, SupervisedModel[I, M], List[(Int, Double, Option[Double])])] = {
    val modelWithOptimizer = model.asTraining.zipOptimizer(optimizerFactory)

    val ec =
      Resource(IO.delay {
        if (prefetchData) {
          val ex1 = Executors.newSingleThreadExecutor()
          val ec1 = ExecutionContext.fromExecutor(ex1)
          val ex2 = Executors.newSingleThreadExecutor()
          val ec2 = ExecutionContext.fromExecutor(ex2)
          (
            Option((ec1, ec2)),
            IO.delay {
              ex1.shutdown()
              ex2.shutdown()
            }
          )
        } else (None, IO.unit)
      })

    ec.use { maybeExecutionContext =>
      def loop(
          epoch: Int,
          lastValidationLoss: Option[Double],
          minValidationLoss: Option[Double],
          minValidationLossModel: Option[(Int, Seq[Tensor])],
          learningCurve: List[(Int, Double, Option[Double])]
      ): IO[
        (Int, SupervisedModel[I, M], List[(Int, Double, Option[Double])])
      ] = {
        val learningRateFactor = learningRateSchedule.learningRateFactor(
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
            trainingLoss <- oneEpoch(
              epoch,
              trainingCallback,
              modelWithOptimizer,
              trainBatchesOverEpoch(),
              logger,
              learningRateFactor,
              prefetchEC = maybeExecutionContext
            )

            _ <-
              if (checkpointFile.isDefined)
                writeCheckpoint(checkpointFile.get, model.module)
              else IO.unit
            maybeValidationLoss <-
              if (
                epoch % validationFrequency == 0 && validationBatchesOverEpoch.isDefined
              )
                validationOneEpoch(
                  model = modelWithOptimizer.model,
                  validationBatches = validationBatchesOverEpoch.get(),
                  validationCallback = validationCallback,
                  logger = logger,
                  epochCount = epoch,
                  minimumCheckpointFile = minimumCheckpointFile,
                  minimumValidationLossSoFar = minValidationLoss
                ).map(Some(_))
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
                (epoch, trainingLoss, maybeValidationLoss) :: learningCurve
            )
          } yield next
        }
      }

      loop(0, None, None, None, Nil)

    }
  }

  def oneEpoch[I, M <: GenericModule[I, Variable]](
      epochCount: Long,
      trainingCallback: TrainingCallback,
      model: ModelWithOptimizer[I, M],
      trainBatches: BatchStream[I],
      logger: Option[Logger],
      learningRateScheduleFactor: Double,
      prefetchEC: Option[(ExecutionContext, ExecutionContext)]
  ): IO[Double] = {

    def processBatch(
        elem: StreamControl[(I, STen)],
        lossAcc: STen
    ): StreamControl[Long] = elem.map { case (sample, target) =>
      val (numInstances, gradients) =
        model.model.addTotalLossAndReturnGradientsAndNumExamples(
          sample,
          target,
          lossAcc
        )

      model.optimizer.step(gradients, learningRateScheduleFactor)
      numInstances

    }

    def simpleLoop(lossAcc: STen, numInstancesAcc: Long): IO[Long] = {
      trainBatches.nextBatch
        .use { batch => IO { processBatch(batch, lossAcc) } }
        .flatMap {
          case EndStream  => IO.pure(numInstancesAcc)
          case EmptyBatch => simpleLoop(lossAcc, numInstancesAcc)
          case NonEmptyBatch(numInstances) =>
            simpleLoop(
              lossAcc,
              numInstances + numInstancesAcc
            )
        }
    }

    def prefetch[A, B](
        fetch: () => Resource[IO, StreamControl[A]],
        transform: (Int, StreamControl[A]) => IO[StreamControl[B]],
        reduce: (B, B) => B,
        zero: B,
        cs1: ContextShift[IO],
        cs2: ContextShift[IO]
    ): IO[B] = {

      def loop(
          counter: Int,
          acc: B,
          prefetching: Fiber[IO, (StreamControl[A], IO[Unit])]
      ): IO[B] =
        for {
          fetched <- prefetching.join
          _ <- cs2.shift
          a = fetched._1
          release = fetched._2
          nextPrefetch <- fetch().allocated.start(cs1)
          done <- transform(
            counter,
            a
          )
          _ <- release
          loopDone <- done match {
            case EndStream  => IO.pure(acc)
            case EmptyBatch => loop(counter, acc, nextPrefetch)
            case NonEmptyBatch(b) =>
              loop(counter + 1, reduce(b, acc), nextPrefetch)
          }
        } yield loopDone

      for {
        _ <- cs2.shift
        f <- IO { fetch().allocated }
        f <- f.start(cs1)
        l <- loop(0, zero, f)
      } yield l

    }

    def prefetchLoop(
        lossAcc: STen,
        ec1: ExecutionContext,
        ec2: ExecutionContext
    ) = {
      val cs1 =
        IO.contextShift(ec1)
      val cs2 =
        IO.contextShift(ec2)
      prefetch[(I, STen), Long](
        fetch = () => trainBatches.nextBatch,
        transform = (_, batch) => IO { processBatch(batch, lossAcc) },
        reduce = (b, acc) => (acc + b),
        zero = 0L,
        cs1 = cs1,
        cs2 = cs2
      )
    }

    val epochLoop = Scope.inResource.use { implicit scope =>
      val lossAcc =
        STen.scalarDouble(0d, model.model.module.state.head._1.options)
      val loopDone = (if (prefetchEC.isDefined) {
                        val (a, b) = prefetchEC.get
                        prefetchLoop(lossAcc, a, b)
                      } else simpleLoop(lossAcc, 0L))

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
      epochCount: Long,
      minimumCheckpointFile: Option[File],
      minimumValidationLossSoFar: Option[Double]
  ): IO[Double] = {
    def loop(
        batchCount: Int,
        totalLoss: STen,
        totalExamples: Long
    ): IO[(STen, Long)] = {
      validationBatches.nextBatch
        .use { elem =>
          IO {
            elem.map { case (validationSample, validationTarget) =>
              val numExamples = model.asEval
                .addTotalLossAndReturnNumExamples(
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

          _ <-
            if (
              minimumCheckpointFile.isDefined && (minimumValidationLossSoFar.isEmpty || minimumValidationLossSoFar.get > validationLoss)
            )
              IO {
                scribe.info(
                  s"Minimum validation loss $validationLoss reached at $epochCount. Writing checkpoint to $minimumCheckpointFile"
                )
              }.flatMap(_ =>
                writeCheckpoint(minimumCheckpointFile.get, model.module)
              )
            else IO.unit
        } yield validationLoss
      }
    }
  }

}
