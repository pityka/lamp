package lamp.data
import lamp.nn._
import cats.effect._
import aten.Tensor
import java.io.File
import scribe.Logger
import Writer.writeCheckpoint
import lamp.autograd.Variable
import lamp.TensorHelpers
import lamp.STen
import lamp.Scope
import scala.concurrent.ExecutionContext
import java.util.concurrent.Executors
object IOLoops {

  def epochs[I, M <: GenericModule[I, Variable]: Load: TrainingMode](
      model: SupervisedModel[I, M],
      optimizerFactory: Seq[(STen, PTag)] => Optimizer,
      trainBatchesOverEpoch: () => BatchStream[I],
      validationBatchesOverEpoch: Option[() => BatchStream[I]],
      epochs: Int,
      trainingBatchCallback: TrainingBatchCallback = TrainingBatchCallback.noop,
      trainingCallback: TrainingCallback = TrainingCallback.noop,
      validationCallback: ValidationCallback = ValidationCallback.noop,
      validationBatchCallback: ValidationBatchCallback =
        ValidationBatchCallback.noop,
      checkpointFile: Option[File] = None,
      minimumCheckpointFile: Option[File] = None,
      validationFrequency: Int = 1,
      logger: Option[Logger] = None,
      returnMinValidationLossModel: Seq[Int] = Nil,
      returnDevice: Option[lamp.Device] = None,
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
          (Option((ec1, ec2)), IO.delay {
            ex1.shutdown()
            ex2.shutdown()
          })
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
        val learningRateFactor = learningRateSchedule.factor(
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
                  val stateOnDevice = state.map { t =>
                    val device =
                      returnDevice.getOrElse(
                        TensorHelpers.device(
                          model.module.state.head._1.value.value
                        )
                      )
                    val t2 = device.to(t)
                    t.release
                    STen.owned(t2)
                  }
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
            val device =
              returnDevice.getOrElse(
                TensorHelpers.device(
                  model.module.state.head._1.value.value
                )
              )
            logger.foreach(_.info(s"Copying model at epoch $epoch"))
            minValidationLossModel.foreach(_._2.foreach(_.release))
            val copiedState =
              model.module.state.map(_._1.value).map { t => device.to(t.value) }

            (epoch, copiedState)
          }

          for {
            trainingLoss <- oneEpoch(
              epoch,
              trainingCallback,
              modelWithOptimizer,
              trainBatchesOverEpoch(),
              trainingBatchCallback,
              logger,
              learningRateFactor,
              prefetchEC = maybeExecutionContext
            )

            _ <- if (checkpointFile.isDefined)
              writeCheckpoint(checkpointFile.get, model.module)
            else IO.unit
            maybeValidationLoss <- if (epoch % validationFrequency == 0 && validationBatchesOverEpoch.isDefined)
              validationOneEpoch(
                model = modelWithOptimizer.model,
                validationBatches = validationBatchesOverEpoch.get(),
                validationCallback = validationCallback,
                validationBatchCallback = validationBatchCallback,
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

            nextMinValidationLoss = if (maybeValidationLoss.isEmpty || !returnMinValidationLossModel
                                          .contains(epoch))
              minValidationLoss
            else if (minValidationLoss.isEmpty) maybeValidationLoss
            else Some(math.min(minValidationLoss.get, maybeValidationLoss.get))

            nextMinValidationLossModel = if (returnMinValidationLossModel
                                               .contains(epoch)) {
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
      trainingBatchCallback: TrainingBatchCallback,
      logger: Option[Logger],
      learningRateScheduleFactor: Double,
      prefetchEC: Option[(ExecutionContext, ExecutionContext)]
  ): IO[Double] = {

    def processBatch(
        batchCount: Int,
        option: Option[(I, STen)]
    ): IO[Option[(Double, Long)]] = {
      val io = option.map {
        case (sample, target) =>
          val (avgLoss, numInstances, gradients) =
            model.model.lossAndGradients(sample, target)
          for {
            _ <- IO {
              trainingBatchCallback(avgLoss, batchCount)
            }

            _ <- IO {
              model.optimizer.step(gradients, learningRateScheduleFactor)
            }
          } yield (avgLoss, numInstances)

      }
      io.map(_.map(Some(_))).getOrElse(IO.pure(None))
    }

    def simpleLoop(batchCount: Int, acc: (Double, Long)): IO[(Double, Long)] = {
      trainBatches.nextBatch
        .use { option => processBatch(batchCount, option) }
        .flatMap {
          case None => IO.pure(acc)
          case Some((avgLoss, numInstances)) =>
            simpleLoop(
              batchCount + 1,
              (avgLoss * numInstances + acc._1, numInstances + acc._2)
            )
        }
    }

    def prefetch[A, B](
        fetch: () => Resource[IO, Option[A]],
        transform: (Int, Option[A]) => IO[Option[B]],
        reduce: (B, B) => B,
        zero: B,
        cs1: ContextShift[IO],
        cs2: ContextShift[IO]
    ): IO[B] = {

      def loop(
          counter: Int,
          acc: B,
          prefetching: Fiber[IO, (Option[A], IO[Unit])]
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
            case None    => IO.pure(acc)
            case Some(b) => loop(counter + 1, reduce(b, acc), nextPrefetch)
          }
        } yield loopDone

      for {
        _ <- cs2.shift
        f <- IO { fetch().allocated }
        f <- f.start(cs1)
        l <- loop(0, zero, f)
      } yield l

    }

    def prefetchLoop(ec1: ExecutionContext, ec2: ExecutionContext) = {
      val cs1 =
        IO.contextShift(ec1)
      val cs2 =
        IO.contextShift(ec2)
      prefetch[(I, STen), (Double, Long)](
        fetch = () => trainBatches.nextBatch,
        transform = (count, batch) => processBatch(count, batch),
        reduce = (b, acc) => (b._1 * b._2 + acc._1, acc._2 + b._2),
        zero = (0d, 0L),
        cs1 = cs1,
        cs2 = cs2
      )
    }

    val epochLoop = (if (prefetchEC.isDefined) {
                       val (a, b) = prefetchEC.get
                       prefetchLoop(a, b)
                     } else simpleLoop(0, (0d, 0L)))

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
      validationBatchCallback: ValidationBatchCallback,
      validationCallback: ValidationCallback,
      logger: Option[Logger],
      epochCount: Long,
      minimumCheckpointFile: Option[File],
      minimumValidationLossSoFar: Option[Double]
  ): IO[Double] = {
    def loop(
        batchCount: Int,
        totalLoss: Double,
        totalExamples: Long
    ): IO[(Double, Long)] = {
      validationBatches.nextBatch
        .use { option =>
          val io = option.map {
            case (validationSample, validationTarget) =>
              model.asEval
                .lossAndOutput(
                  validationSample,
                  validationTarget
                )
                .use {
                  case (validationLoss, validationOutput, numExamples) =>
                    for {
                      _ <- IO {
                        validationBatchCallback(
                          validationOutput,
                          validationTarget,
                          validationLoss,
                          epochCount,
                          batchCount
                        )
                      }

                    } yield (validationLoss * numExamples, numExamples)
                }
          }
          io.map(_.map(Some(_))).getOrElse(IO.pure(None))
        }
        .flatMap {
          case None => IO.pure((totalLoss, totalExamples))
          case Some((lossInBatch, examples)) =>
            loop(
              batchCount + 1,
              totalLoss + lossInBatch,
              totalExamples + examples
            )
        }

    }

    loop(0, 0d, 0L).flatMap {
      case (totalLoss, totalExamples) =>
        val validationLoss = totalLoss / totalExamples
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

          _ <- if (minimumCheckpointFile.isDefined && (minimumValidationLossSoFar.isEmpty || minimumValidationLossSoFar.get > validationLoss))
            IO {
              scribe.info(
                s"Minimum validation loss $validationLoss reached at $epochCount. Writing checkpoint to $minimumCheckpointFile"
              )
            }.flatMap(_ =>
              writeCheckpoint(minimumCheckpointFile.get, model.module)
            )
          else IO.unit
        } yield totalLoss / totalExamples
    }
  }

}
