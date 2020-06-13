package lamp.data
import lamp.nn._
import cats.effect._
import aten.Tensor
import lamp.syntax
import lamp.util.NDArray
import java.io.FileOutputStream
import java.io.File
import org.saddle.scalar.ScalarTagDouble
import scribe.Logger
object IOLoops {

  def writeCheckpoint(file: File, model: Module) = {
    val channel = Resource.make(IO {
      val fis = new FileOutputStream(file, false)
      fis.getChannel
    })(v => IO { v.close })
    channel
      .use { channel =>
        IO {
          Writer.writeTensorsIntoChannel(
            model.state
              .map(v => (ScalarTagDouble, v._1.value)),
            channel
          )
        }
      }
  }

  def epochs(
      model: SupervisedModel,
      optimizerFactory: Seq[(Tensor, PTag)] => Optimizer,
      trainBatchesOverEpoch: () => BatchStream,
      validationBatchesOverEpoch: () => BatchStream,
      epochs: Int,
      trainingCallback: TrainingCallback,
      validationCallback: ValidationCallback,
      checkpointFile: Option[File],
      minimumCheckpointFile: Option[File],
      checkpointFrequency: Int,
      logger: Option[Logger] = None
  ): IO[SupervisedModel] = {
    val modelWithOptimizer = model.asTraining.zipOptimizer(optimizerFactory)

    def loop(
        epoch: Int,
        currentValidation: BatchStream,
        minValidationLoss: Option[Double]
    ): IO[SupervisedModel] =
      if (epoch >= epochs) IO.pure(modelWithOptimizer.model)
      else {
        for {
          _ <- oneEpoch(
            modelWithOptimizer,
            trainBatchesOverEpoch(),
            trainingCallback,
            logger
          )
          maybeValidationLoss <- runValidation(
            modelWithOptimizer.model,
            currentValidation.nextBatch,
            validationCallback,
            epoch,
            minValidationLoss,
            checkpointFile,
            minimumCheckpointFile,
            checkpointFrequency,
            logger
          )
          replaceValidation = maybeValidationLoss.isEmpty
          nextMinValidationLoss = if (maybeValidationLoss.isEmpty)
            minValidationLoss
          else if (minValidationLoss.isEmpty) maybeValidationLoss
          else Some(math.min(minValidationLoss.get, maybeValidationLoss.get))
          next <- loop(
            epoch + 1,
            if (replaceValidation) validationBatchesOverEpoch()
            else currentValidation,
            nextMinValidationLoss
          )
        } yield next
      }

    loop(0, validationBatchesOverEpoch(), None)
  }

  def runValidation(
      model: SupervisedModel,
      validationBatch: Resource[IO, Option[(Tensor, Tensor)]],
      validationCallback: ValidationCallback,
      epochCount: Int,
      minimumValidationLossSoFar: Option[Double],
      checkpointFile: Option[File],
      minimumCheckpointFile: Option[File],
      checkpointFrequency: Int,
      logger: Option[Logger]
  ): IO[Option[Double]] = {
    validationBatch
      .use { option =>
        val io = option.map {
          case (validationSample, validationTarget) =>
            model.asEval
              .lossAndOutput(
                validationSample,
                validationTarget
              )
              .use {
                case (validationLoss, validationOutput) =>
                  for {
                    _ <- IO {
                      logger.foreach(
                        _.info(
                          s"Validation loss at epoch $epochCount: $validationLoss"
                        )
                      )
                    }
                    _ <- IO {
                      validationCallback(
                        validationOutput,
                        validationTarget,
                        validationLoss,
                        epochCount
                      )
                    }

                    _ <- if (checkpointFile.isDefined && (epochCount % checkpointFrequency == 0))
                      writeCheckpoint(checkpointFile.get, model.module)
                    else IO.unit
                    _ <- if (minimumCheckpointFile.isDefined && (minimumValidationLossSoFar.isEmpty || minimumValidationLossSoFar.get > validationLoss))
                      IO {
                        scribe.info(
                          s"Minimum validation loss $validationLoss reached at $epochCount. Writing checkpoint to $checkpointFile"
                        )
                      }.flatMap(_ =>
                        writeCheckpoint(minimumCheckpointFile.get, model.module)
                      )
                    else IO.unit
                  } yield validationLoss
              }

        }
        io.map(_.map(Some(_))).getOrElse(IO.pure(None))
      }
  }

  def oneEpoch(
      model: ModelWithOptimizer,
      trainBatches: BatchStream,
      trainingCallback: TrainingCallback,
      logger: Option[Logger]
  ): IO[Unit] = {

    def loop(batchCount: Int): IO[Unit] = {
      trainBatches.nextBatch
        .use { option =>
          val io = option.map {
            case (sample, target) =>
              model.model.lossAndGradientsAndOutput(sample, target).use {
                case (loss, gradients, output) =>
                  for {
                    _ <- IO {
                      trainingCallback(loss, batchCount, output, target)
                    }
                    _ <- IO {
                      logger.foreach(
                        _.info(s"Training loss at batch $batchCount: $loss")
                      )
                    }
                    _ <- IO { model.optimizer.step(gradients) }
                  } yield ()

              }
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
}
