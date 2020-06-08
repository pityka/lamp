package lamp.data
import lamp.nn._
import cats.effect._
import aten.Tensor

object IOLoops {

  def epochs(
      model: SupervisedModel,
      optimizerFactory: Seq[(Tensor, PTag)] => Optimizer,
      trainBatchesOverEpoch: () => BatchStream,
      validationBatchesOverEpoch: () => BatchStream,
      epochs: Int,
      trainingCallback: TrainingCallback,
      validationCallback: ValidationCallback
  ): IO[SupervisedModel] = {
    val modelWithOptimizer = model.asTraining.zipOptimizer(optimizerFactory)

    def loop(epoch: Int, currentValidation: BatchStream): IO[SupervisedModel] =
      if (epoch >= epochs) IO.pure(modelWithOptimizer.model)
      else {
        for {
          _ <- oneEpoch(
            modelWithOptimizer,
            trainBatchesOverEpoch(),
            trainingCallback
          )
          replaceValidation <- runValidation(
            modelWithOptimizer.model,
            currentValidation.nextBatch,
            validationCallback,
            epoch
          )
          next <- loop(
            epoch + 1,
            if (replaceValidation) validationBatchesOverEpoch()
            else currentValidation
          )
        } yield next
      }

    loop(0, validationBatchesOverEpoch())
  }

  def runValidation(
      model: SupervisedModel,
      validationBatch: Resource[IO, Option[(Tensor, Tensor)]],
      validationCallback: ValidationCallback,
      epochCount: Int
  ) = {
    validationBatch
      .use { option =>
        IO {
          option.map {
            case (validationSample, validationTarget) =>
              val (validationLoss, validationOutput) =
                model.asEval.lossAndOutput(
                  validationSample,
                  validationTarget
                )
              validationCallback(
                validationOutput,
                validationTarget,
                validationLoss,
                epochCount
              )

              validationOutput.release

          }
        }
      }
      .map(_.isEmpty)
  }

  def oneEpoch(
      model: ModelWithOptimizer,
      trainBatches: BatchStream,
      trainingCallback: TrainingCallback
  ): IO[Unit] = {

    def loop(batchCount: Int): IO[Unit] = {
      trainBatches.nextBatch
        .use { option =>
          IO {
            option.map {
              case (sample, target) =>
                val (loss, gradients, output) =
                  model.model.lossAndGradientsAndOutput(sample, target)
                trainingCallback(loss, batchCount, output, target)
                model.optimizer.step(gradients)
                output.release
            }
          }
        }
        .flatMap {
          case None => IO.unit
          case _    => loop(batchCount + 1)
        }

    }

    loop(0)
  }
}
