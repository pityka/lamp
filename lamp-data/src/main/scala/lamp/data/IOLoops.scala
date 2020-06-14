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

  def writeCheckpoint[T](file: File, model: StatefulModule[T]) = {
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

  def epochs[ST](
      model: SupervisedModel[ST],
      optimizerFactory: Seq[(Tensor, PTag)] => Optimizer,
      trainBatchesOverEpoch: () => BatchStream,
      validationBatchesOverEpoch: () => BatchStream,
      epochs: Int,
      trainingCallback: TrainingCallback,
      validationCallback: ValidationCallback,
      checkpointFile: Option[File],
      minimumCheckpointFile: Option[File],
      checkpointFrequency: Int,
      logFrequency: Int = 1,
      validationFrequency: Int = 1,
      logger: Option[Logger] = None
  ): IO[SupervisedModel[ST]] = {
    val modelWithOptimizer = model.asTraining.zipOptimizer(optimizerFactory)

    def loop(
        epoch: Int,
        minValidationLoss: Option[Double]
    ): IO[SupervisedModel[ST]] =
      if (epoch >= epochs) IO.pure(modelWithOptimizer.model)
      else {
        for {
          _ <- oneEpoch(
            modelWithOptimizer,
            trainBatchesOverEpoch(),
            trainingCallback,
            logger,
            logFrequency
          )
          maybeValidationLoss <- if (epoch % validationFrequency == 0)
            validationOneEpoch(
              model = modelWithOptimizer.model,
              validationBatches = validationBatchesOverEpoch(),
              validationCallback = validationCallback,
              logger = logger,
              validationLogFrequency = logFrequency,
              epochCount = epoch
            ).map(Some(_))
          else IO.pure(None)

          nextMinValidationLoss = if (maybeValidationLoss.isEmpty)
            minValidationLoss
          else if (minValidationLoss.isEmpty) maybeValidationLoss
          else Some(math.min(minValidationLoss.get, maybeValidationLoss.get))
          next <- loop(
            epoch + 1,
            nextMinValidationLoss
          )
        } yield next
      }

    loop(0, None)
  }

  def oneEpoch[T](
      model: ModelWithOptimizer[T],
      trainBatches: BatchStream,
      trainingCallback: TrainingCallback,
      logger: Option[Logger],
      trainLogFrequency: Int
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
                        s"Training loss at batch $batchCount: $loss (exp: ${math.exp(loss)})"
                      )
                    )
                  }
                }
                _ <- IO { model.optimizer.step(gradients) }
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
  def validationOneEpoch[T](
      model: SupervisedModel[T],
      validationBatches: BatchStream,
      validationCallback: ValidationCallback,
      logger: Option[Logger],
      validationLogFrequency: Int,
      epochCount: Long
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
                      _ <- IO {
                        if (batchCount % validationLogFrequency == 0) {
                          logger.foreach(
                            _.info(
                              s"Validation loss at batch $batchCount in epoch $epochCount over $numExamples examples: $validationLoss (exp: ${math
                                .exp(validationLoss)})"
                            )
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
        IO {
          logger.foreach(
            _.info(
              s"Avg validation loss in epoch $epochCount over $totalExamples examples: ${totalLoss / totalExamples}"
            )
          )
          totalLoss / totalExamples
        }
    }
  }

}
