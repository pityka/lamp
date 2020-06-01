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
      epochs: Int
  )(callbackTraining: Double => Unit)(
      callbackOnValidationOutputAndTarget: (Tensor, Tensor, Double) => Unit
  ): IO[SupervisedModel] = {
    val modelWithOptimizer = model.asTraining.zipOptimizer(optimizerFactory)

    def loop(epoch: Int, currentValidation: BatchStream): IO[SupervisedModel] =
      if (epoch >= epochs) IO.pure(modelWithOptimizer.model)
      else {
        for {
          _ <- oneEpoch(modelWithOptimizer, trainBatchesOverEpoch())(
            callbackTraining
          )
          replaceValidation <- runValidation(
            modelWithOptimizer.model,
            currentValidation.nextBatch
          )(callbackOnValidationOutputAndTarget)
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
      validationBatch: Resource[IO, Option[(Tensor, Tensor)]]
  )(callbackOnValidationOutputAndTarget: (Tensor, Tensor, Double) => Unit) = {
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
              callbackOnValidationOutputAndTarget(
                validationOutput,
                validationTarget,
                validationLoss
              )

              validationOutput.release

          }
        }
      }
      .map(_.isEmpty)
  }

  def oneEpoch(
      model: ModelWithOptimizer,
      trainBatches: BatchStream
  )(callback: Double => Unit): IO[Unit] = {

    def loop(): IO[Unit] = {
      trainBatches.nextBatch
        .use { option =>
          IO {
            option.map {
              case (sample, target) =>
                val (loss, gradients) =
                  model.model.lossAndGradients(sample, target)
                callback(loss)
                model.optimizer.step(gradients)
            }
          }
        }
        .flatMap {
          case None => IO.unit
          case _    => loop()
        }

    }

    loop()
  }
}
