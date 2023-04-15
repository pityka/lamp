package lamp.example.lm

import lamp._
import lamp.nn._
import lamp.data._
import java.io.File
import cats.effect.IO
import lamp.data.distributed.LoopStateWithModelAndOptimizerData

object DistributedTrain {
  def train(
      config: CliConfig,
      trainTokens: STen,
      validTokens: STen
  )(implicit scope: Scope): IO[Unit] = Scope.bracket(scope) { implicit scope =>
    scribe.info(
      s"Distributed training rank: ${config.rank} nrank:${config.nranks} gpu:${config.gpu}"
    )
    val device = CudaDevice(config.gpu)
    val model = Model.allocateModel(device)

    val actorSystem = akka.actor.ActorSystem(
      name = s"lm-${config.rank}",
      config = Some(
        com.typesafe.config.ConfigFactory.parseString(
          s"""
          
akka {
  jvm-shutdown-hooks = off
  actor {
    provider = remote
  }
  remote {
    artery {
      transport = tcp
      canonical.hostname = "${config.myAddress}"
      canonical.port = ${config.myPort}
    }
  }
}
                """
        )
      )
    )

    val trainEpochs = () =>
      lamp.data.languagemodel
        .autoregressiveMinibatchesFromCorpus(
          minibatchSize = config.trainBatchSize,
          numBatches = config.numBatchesPerEpoch,
          corpus = trainTokens,
          blockLength = Model.contextLength
        )
        .withoutEmptyBatches
        .everyNth(n = config.nranks, offset = config.rank)

    val validEpochs = () =>
      lamp.data.languagemodel
        .autoregressiveMinibatchesFromCorpus(
          minibatchSize = config.trainBatchSize,
          numBatches = config.numBatchesPerEpoch,
          corpus = validTokens,
          blockLength = Model.contextLength
        )
        .withoutEmptyBatches
        .everyNth(n = config.nranks, offset = config.rank)

    val program = if (config.rank == 0) {
      // this branch is executed on rank=0 process

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

      val checkpointedState = config.checkpointSave.flatMap { state =>
        if (new File(state).canRead()) {
          Some(
            {
              val sl = StateIO
                .readFromFile(new File(state), device)
                .asInstanceOf[SimpleLoopState]
              lamp.data.distributed.LoopStateWithModelAndOptimizerData(sl)
            }
          )
        } else None
      }

      val comm = new lamp.distributed.akka.AkkaCommunicationServer(
        actorSystem
      )
      lamp.data.distributed.driveDistributedTraining(
        nranks = config.nranks,
        gpu = config.gpu,
        controlCommunication = comm,
        model = model,
        optimizerFactory = optimizer,
        trainBatches = trainEpochs,
        validationBatches = validEpochs,
        maxEpochs = config.epochs,
        initState = checkpointedState,
        checkpointState =
          Some((state: LoopStateWithModelAndOptimizerData, _: Unit) =>
            config.checkpointSave
              .map { file =>
                lamp.data.distributed.LoopState
                  .stateToFile(new File(file))(state)
              }
              .getOrElse(IO.unit)
          )
      )

    } else {
      // this branch is executed on rank>0 processes (in parallel/lockstep because rank=0 waits for rendez-vous)
      import scala.concurrent.duration._
      val comm = new lamp.distributed.akka.AkkaCommunicationClient(
        actorSystem,
        config.rootAddress,
        config.rootPort,
        "lm-0",
        600 seconds
      )

      lamp.data.distributed.followDistributedTraining(
        rank = config.rank,
        nranks = config.nranks,
        gpu = config.gpu,
        controlCommunication = comm,
        model = model,
        trainBatches = trainEpochs,
        validationBatches = validEpochs
      )

    }

    program.flatMap { _ =>
      IO.fromFuture(IO(actorSystem.terminate())) *> IO.unit

    }

  }
}
