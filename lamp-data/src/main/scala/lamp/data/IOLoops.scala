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
object IOLoops {

  def epochs[I, M <: GenericModule[I, Variable]: Load: TrainingMode](
      model: SupervisedModel[I, M],
      optimizerFactory: Seq[(STen, PTag)] => Optimizer,
      trainBatchesOverEpoch: () => BatchStream[I],
      validationBatchesOverEpoch: Option[() => BatchStream[I]],
      epochs: Int,
      trainingCallback: TrainingCallback = TrainingCallback.noop,
      validationCallback: ValidationCallback = ValidationCallback.noop,
      checkpointFile: Option[File] = None,
      minimumCheckpointFile: Option[File] = None,
      logFrequency: Int = 1,
      validationFrequency: Int = 1,
      logger: Option[Logger] = None,
      returnMinValidationLossModel: Seq[Int] = Nil,
      returnDevice: Option[lamp.Device] = None,
      learningRateSchedule: LearningRateSchedule = LearningRateSchedule.noop
  ): IO[(Int, SupervisedModel[I, M])] = {
    val modelWithOptimizer = model.asTraining.zipOptimizer(optimizerFactory)

    def loop(
        epoch: Int,
        lastValidationLoss: Option[Double],
        minValidationLoss: Option[Double],
        minValidationLossModel: Option[(Int, Seq[Tensor])]
    ): IO[(Int, SupervisedModel[I, M])] =
      if (epoch >= epochs)
        IO.pure {
          modelWithOptimizer.optimizer.release()
          minValidationLossModel match {
            case None => (epoch - 1, modelWithOptimizer.model)
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
              (epochOfMinValidation, modelWithOptimizer.model)
          }
        }
      else {

        def copyModel = {
          logger.foreach(_.info(s"Copying model at epoch $epoch"))
          minValidationLossModel.foreach(_._2.foreach(_.release))
          val copiedState =
            model.module.state.map(_._1.value).map { t => lamp.CPU.to(t.value) }

          (epoch, copiedState)
        }

        for {
          _ <- oneEpoch(
            modelWithOptimizer,
            trainBatchesOverEpoch(),
            trainingCallback,
            logger,
            logFrequency,
            learningRateSchedule
              .factor(epoch = epoch, lastValidationLoss = lastValidationLoss)
          )
          _ <- if (checkpointFile.isDefined)
            writeCheckpoint(checkpointFile.get, model.module)
          else IO.unit
          maybeValidationLoss <- if (epoch % validationFrequency == 0 && validationBatchesOverEpoch.isDefined)
            validationOneEpoch(
              model = modelWithOptimizer.model,
              validationBatches = validationBatchesOverEpoch.get(),
              validationCallback = validationCallback,
              logger = logger,
              validationLogFrequency = logFrequency,
              epochCount = epoch,
              minimumCheckpointFile = minimumCheckpointFile,
              minimumValidationLossSoFar = minValidationLoss
            ).map(Some(_))
          else IO.pure(None)

          nextMinValidationLoss = if (maybeValidationLoss.isEmpty)
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
            minValidationLossModel = nextMinValidationLossModel
          )
        } yield next
      }

    loop(0, None, None, None)
  }

  def oneEpoch[I, M <: GenericModule[I, Variable]](
      model: ModelWithOptimizer[I, M],
      trainBatches: BatchStream[I],
      trainingCallback: TrainingCallback,
      logger: Option[Logger],
      trainLogFrequency: Int,
      learningRateScheduleFactor: Double
  ): IO[Unit] = {

    def loop(batchCount: Int): IO[Unit] = {
      trainBatches.nextBatch
        .use { option =>
          val io = option.map {
            case (sample, target) =>
              val (loss, gradients) =
                model.model.lossAndGradients(sample, target)

              for {
                _ <- IO {
                  if (batchCount % trainLogFrequency == 0) {
                    trainingCallback(loss, batchCount)
                  }
                }
                _ <- IO {
                  if (batchCount % trainLogFrequency == 0) {
                    logger.foreach(
                      _.info(
                        s"Training loss in batch $batchCount: $loss (exp: ${math.exp(loss)})"
                      )
                    )
                  }
                }
                _ <- IO {
                  model.optimizer.step(gradients, learningRateScheduleFactor)
                }
              } yield ()

          }
          io.map(_.map(Some(_))).getOrElse(IO.pure(None))
        }
        .flatMap {
          case None => IO.unit
          case _    => loop(batchCount + 1)
        }

    }

    loop(0)
  }
  def validationOneEpoch[I, M <: GenericModule[I, Variable]](
      model: SupervisedModel[I, M],
      validationBatches: BatchStream[I],
      validationCallback: ValidationCallback,
      logger: Option[Logger],
      validationLogFrequency: Int,
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
                        if (batchCount % validationLogFrequency == 0) {
                          validationCallback(
                            validationOutput,
                            validationTarget,
                            validationLoss,
                            epochCount
                          )
                        }
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
