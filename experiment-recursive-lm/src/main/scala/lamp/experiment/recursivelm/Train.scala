package lamp.experiment.recursivelm

import lamp._
import lamp.nn._
import lamp.data._
import java.io.File
import cats.effect.IO

object Train {
  def train(
      config: CliConfig,
      trainDocuments: Array[STen],
      validDocuments: Array[STen]
  )(implicit scope: Scope): IO[Unit] = Scope.bracket(scope) { implicit scope =>
    val device =
      if (config.gpus.nonEmpty) CudaDevice(config.gpus.head) else CPU

    val model = Model.allocateModel(device)

    val extraModels = config.gpus.drop(1).map { deviceNum =>
      val device = CudaDevice(deviceNum)
      Model.allocateModel(device)
    }

    val checkpointedState = config.checkpointSave.flatMap { state =>
      if (new File(state).canRead()) {
        Some(
          StateIO
            .readFromFile(new File(state), device)
            .asInstanceOf[SimpleLoopState]
        )
      } else None
    }

    val trainEpochs = (_: IOLoops.TrainingLoopContext) =>
      lamp.experiment.recursivelm.model.DataLoader.minibatchesFromDocuments(
        minibatchSize = config.trainBatchSize,
        numBatches = config.numBatchesPerEpoch,
        documents = trainDocuments,
        blockLength = Model.contextLength
      )
    val validEpochs = (_: IOLoops.TrainingLoopContext) =>
      lamp.experiment.recursivelm.model.DataLoader.minibatchesFromDocuments(
        minibatchSize = config.trainBatchSize,
        numBatches = config.numBatchesPerEpoch,
        documents = validDocuments,
        blockLength = Model.contextLength
      )

    val optimizer = AdamW.factory(
      weightDecay = lamp.nn.DependentHyperparameter(0d) {
        case TransformerEncoderBlock.Weights1 => config.weightDecay
        case TransformerEncoderBlock.Weights2 => config.weightDecay
        case MultiheadAttention.WeightsK      => config.weightDecay
        case MultiheadAttention.WeightsQ      => config.weightDecay
        case MultiheadAttention.WeightsV      => config.weightDecay
        case MultiheadAttention.WeightsO      => config.weightDecay
      },
      learningRate = simple(config.learningRate),
      beta2 = simple(config.beta2),
      clip = Some(1d)
    )

    IOLoops
      .epochs(
        model = model,
        optimizerFactory = optimizer,
        trainBatchesOverEpoch = trainEpochs,
        validationBatchesOverEpoch = Some(validEpochs),
        epochs = config.epochs,
        initState = checkpointedState,
        logger = Some(scribe.Logger("training")),
        validationFrequency = 1,
        dataParallelModels = extraModels,
        accumulateGradientOverNBatches = config.gradientAccumSteps,
        checkpointState = Some((state: LoopState, _: Unit) =>
          config.checkpointSave
            .map { file =>
              StateIO.stateToFile(new File(file))(state)
            }
            .getOrElse(IO.unit)
        )
      )
      .map { _ =>
        scribe.info("Training done.")

      }

  }
}
