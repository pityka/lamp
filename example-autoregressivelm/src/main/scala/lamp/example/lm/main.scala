package lamp.example.lm

import lamp._

import cats.effect.IO
import cats.effect.ExitCode
import cats.effect.IOApp

object Main extends IOApp {
  scribe.info("Logger start")

  override def run(args: List[String]): IO[ExitCode] =
    CliParser.runCli(args.toList) {
      case config if config.extend.isEmpty =>
        scribe.info(s"Config: $config")
        Scope.inResource.use(scope =>
          for {
            corpora <- Util.prepareCorpora(config)(scope)
            _ <- Train.train(config, corpora._1, corpora._2)(scope)
          } yield ExitCode(0)
        )

      case config =>
        scribe.info(s"Config: $config")
        scribe.info(s"Inference mode. Extending '${config.extend.get}'")

        for {
          codec <- Model.codecFactory.readFromFile(config.bpeFile.get)
          _ <- Scope.inResource.use(scope =>
            Inference.inference(config, codec)(scope)
          )

        } yield ExitCode(0)

    }

}

// if (config.distributed) {
//         Scope.root { implicit scope =>
//           scribe.info(
//             s"Distributed training rank: ${config.rank} nrank:${config.nranks} gpu:${config.gpu}"
//           )
//           val device = CudaDevice(config.gpu)
//           val model = allocateModel(device)

//           val actorSystem = akka.actor.ActorSystem(
//             name = s"lm-${config.rank}",
//             config = Some(
//               com.typesafe.config.ConfigFactory.parseString(
//                 s"""
// akka {
//   actor {
//     provider = remote
//   }
//   remote {
//     artery {
//       transport = tcp
//       canonical.hostname = "${config.myAddress}"
//       canonical.port = ${config.myPort}
//     }
//   }
// }
//                 """
//               )
//             )
//           )

//           val trainEpochs = () =>
//             lamp.data.languagemodel
//               .autoregressiveMinibatchesFromCorpus(
//                 minibatchSize = config.trainBatchSize,
//                 numBatches = config.numBatchesPerEpoch,
//                 corpus = trainCorpus,
//                 blockLength = contextLength
//               )
//               .withoutEmptyBatches
//               .everyNth(n = config.nranks, offset = config.rank)

//           val validEpochs = () =>
//             lamp.data.languagemodel
//               .autoregressiveMinibatchesFromCorpus(
//                 minibatchSize = config.trainBatchSize,
//                 numBatches = config.numBatchesPerEpoch,
//                 corpus = validCorpus,
//                 blockLength = contextLength
//               )
//               .withoutEmptyBatches
//               .everyNth(n = config.nranks, offset = config.rank)

//           val program = if (config.rank == 0) {

//             val optimizer = AdamW.factory(
//               weightDecay = simple(config.weightDecay),
//               learningRate = simple(config.learningRate),
//               beta2 = simple(config.beta2),
//               clip = Some(1d)
//             )

//             // val checkpointedState = config.checkpointLoad.map { state =>
//             //   StateIO
//             //     .readFromFile(new File(state), device)
//             //     .asInstanceOf[SimpleLoopState]
//             // }

//             val comm = new lamp.distributed.akka.AkkaCommunicationServer(
//               actorSystem
//             )
//             lamp.data.distributed.driveDistributedTraining(
//               nranks = config.nranks,
//               gpu = config.gpu,
//               controlCommunication = comm,
//               model = model,
//               optimizerFactory = optimizer,
//               trainBatches = trainEpochs,
//               validationBatches = validEpochs,
//               maxEpochs = config.epochs
//               // initState = checkpointedState,
//               // checkpointState = Some((state: LoopState, _: Unit) =>
//               //   config.checkpointSave
//               //     .map { file =>
//               //       StateIO.stateToFile(new File(file))(state)
//               //     }
//               //     .getOrElse(IO.unit)
//               // )
//             )

//           } else {
//             import scala.concurrent.duration._
//             val comm = new lamp.distributed.akka.AkkaCommunicationClient(
//               actorSystem,
//               config.rootAddress,
//               config.rootPort,
//               "cifar-0",
//               600 seconds
//             )

//             lamp.data.distributed.followDistributedTraining(
//               rank = config.rank,
//               nranks = config.nranks,
//               gpu = config.gpu,
//               controlCommunication = comm,
//               model = model,
//               trainBatches = trainEpochs,
//               validationBatches = validEpochs
//             )

//           }

//           program.unsafeRunSync()
//           actorSystem.terminate()
//           ()
//         }
