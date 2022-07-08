package lamp.example.cifar

import java.io.File

import cats.effect.Resource
import cats.effect.IO
import lamp.CudaDevice
import lamp.CPU
import lamp.data.BatchStream
import lamp.data.IOLoops
import lamp.nn.SupervisedModel
import lamp.nn.AdamW
import lamp.nn.simple
import lamp.nn.LossFunctions
import lamp.data.BufferedImageHelper

import lamp.data.Reader
import lamp.DoublePrecision
import lamp.FloatingPointPrecision
import lamp.HalfPrecision
import lamp.SinglePrecision
import lamp.Scope
import lamp.STen
import lamp.onnx.VariableInfo
import lamp.data.{EndStream, EmptyBatch, NonEmptyBatch}
import cats.effect.unsafe.implicits.global
import lamp.data.DistributedDataParallel
import com.typesafe.config.ConfigFactory
import lamp.NcclUniqueId
import akka.actor.Props
import scala.concurrent.duration._
import aten.NcclComm
import cats.effect.kernel.Deferred
import akka.pattern.ask
import cats.effect.std.Queue
import akka.actor.ActorRef
import lamp.nn.Optimizer

object Cifar {
  def loadImageFile(
      file: File,
      numImages: Int
  )(implicit scope: Scope) = {
    Scope { implicit scope =>
      val all = STen
        .fromFile(
          file.getAbsolutePath(),
          offset = 0,
          length = numImages * 3074,
          scalarTypeByte = 0,
          pin = false
        )
        .view(numImages.toLong, -1)
      val label2 = all.select(1, 1).castToLong
      val images0 = all.slice(1, 2, 3074, 1).view(-1, 3, 32, 32)
      val images = images0.castToFloat
      println(images)
      (label2, images)
    }

  }
}

case class CliConfig(
    trainData: String = "",
    testData: String = "",
    gpu: Int = 0,
    rank: Int = 0,
    nranks: Int = 0,
    rootAddress: String = "",
    rootPort: Int = 28888,
    myAddress: String = "",
    myPort: Int = 28888,
    trainBatchSize: Int = 32,
    testBatchSize: Int = 32,
    epochs: Int = 10,
    learningRate: Double = 0.001,
    dropout: Double = 0.0
)

object Train extends App {
  scribe.info("Logger start")
  import scopt.OParser
  val builder = OParser.builder[CliConfig]
  val parser1 = {
    import builder._
    OParser.sequence(
      opt[String]("train-data")
        .action((x, c) => c.copy(trainData = x))
        .text("path to cifar100 binary train data")
        .required(),
      opt[String]("test-data")
        .action((x, c) => c.copy(testData = x))
        .text("path to cifar100 binary test data")
        .required(),
      opt[Int]("gpu").action((v, c) => c.copy(gpu = v)),
      opt[Int]("rank").action((v, c) => c.copy(rank = v)),
      opt[Int]("nranks").action((v, c) => c.copy(nranks = v)),
      opt[String]("root-address").action((v, c) => c.copy(rootAddress = v)),
      opt[Int]("root-port").action((v, c) => c.copy(rootPort = v)),
      opt[String]("my-address").action((v, c) => c.copy(myAddress = v)),
      opt[Int]("my-port").action((v, c) => c.copy(myPort = v)),
      opt[Int]("batch-train").action((x, c) => c.copy(trainBatchSize = x)),
      opt[Int]("batch-test").action((x, c) => c.copy(testBatchSize = x)),
      opt[Int]("epochs").action((x, c) => c.copy(epochs = x)),
      opt[Double]("learning-rate").action((x, c) => c.copy(learningRate = x)),
      opt[Double]("dropout").action((x, c) => c.copy(dropout = x))
    )

  }

  OParser.parse(parser1, args, CliConfig()) match {
    case Some(config) =>
      scribe.info(s"Config: $config")
      Scope.root { implicit scope =>
        val device = CudaDevice(config.gpu)

        val tensorOptions = device.options(SinglePrecision)
        val model = {
          val numClasses = 100
          val classWeights = STen.ones(List(numClasses), tensorOptions)
          val net =
            Cnn.resnet(numClasses, config.dropout, tensorOptions)

          scribe.info("Learnable parametes: " + net.learnableParameters)
          SupervisedModel(
            net,
            LossFunctions.NLL(numClasses, classWeights),
            printMemoryAllocations = false
          )
        }

        val (trainTarget, trainFullbatch) =
          Cifar.loadImageFile(
            new File(config.trainData),
            50000
          )
        val (testTarget, testFullbatch) =
          Cifar.loadImageFile(
            new File(config.testData),
            10000
          )

        scribe.info(
          s"Loaded full batch data. Train shape: ${trainFullbatch.shape}. Test shape ${testFullbatch.shape}"
        )

        val rng = new scala.util.Random
        val trainBatches = () =>
          BatchStream
            .minibatchesFromFull(
              config.trainBatchSize,
              true,
              trainFullbatch,
              trainTarget,
              rng
            )
            .withoutEmptyBatches
            .everyNth(n = config.nranks, offset = config.rank)

        // val testEpochs = (_: IOLoops.TrainingLoopContext) =>
        //   BatchStream
        //     .minibatchesFromFull(
        //       config.testBatchSize,
        //       true,
        //       testFullbatch,
        //       testTarget,
        //       rng
        //     )
        //     .withoutEmptyBatches
        //     .everyNth(n = config.nranks, offset = config.rank)

        val actorSystem = akka.actor.ActorSystem(
          name = s"cifar-${config.rank}",
          config = Some(
            ConfigFactory.parseString(
              s"""
akka {
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

        val program = if (config.rank == 0) {
          val uid = NcclUniqueId()
          actorSystem.actorOf(Props(new UniqueIdServer(uid)), "uid")
          val ranks = actorSystem.actorOf(Props(new RankRepository), "ranks")

          val ncclComm =
            IO.blocking {
              scribe.info("Waiting for clique")
              STen.ncclInitComm(config.nranks, config.rank, config.gpu, uid)
            }

          val otherRanks = IO
            .fromFuture(IO(ranks.ask("get")(5 seconds)))
            .map(_.asInstanceOf[List[ActorRef]])

          val optimizer = IO {
            val optimizerFactory = AdamW.factory(
              weightDecay = simple(0.00),
              learningRate = simple(config.learningRate)
            )
            optimizerFactory(
              model.module.parameters.map(v => (v._1.value, v._2))
            )
          }

          def trainEpoch(optimizer: Optimizer, comm: NcclComm): IO[Double] =
            DistributedDataParallel.oneEpoch(
              model = model,
              stepOptimizerFn = Some({ gradients =>
                optimizer.step(gradients, 1d)
              }),
              trainBatches = trainBatches(),
              logger = Some(scribe.Logger()),
              accumulateGradientOverNBatches = 1,
              ncclComm = comm,
              rootRank = 0,
              device = device,
              forwardOnly = false
            )

          def trainEpochOnCompleteClick(
              otherRanks: List[ActorRef],
              optimizer: Optimizer,
              comm: NcclComm
          ) =
            IO(
              otherRanks.foreach(_ ! "train")
            ) *> trainEpoch(optimizer, comm)

          for {
            ncclComm <- ncclComm
            otherRanks <- otherRanks
            optimizer <- optimizer
            r <- DistributedDataParallel.epochs[Unit](
              maxEpochs = config.epochs,
              checkpointState = None,
              trainEpoch = (_: Unit) =>
                trainEpochOnCompleteClick(otherRanks, optimizer, ncclComm),
              validationEpoch = None,
              saveModel = IO.unit
            )
            _ <- IO(otherRanks.foreach(_ ! "stop"))
          } yield r

        } else {

          def getCommandQueue = for {
            q <- Queue.bounded[IO, String](1)
            rankRepository <- IO
              .fromFuture(
                IO(
                  actorSystem
                    .actorSelection(
                      s"akka://cifar-0@${config.rootAddress}:${config.rootPort}/user/ranks"
                    )
                    .resolveOne(60 seconds)
                )
              )
            _ <- IO {
              actorSystem.actorOf(
                Props(
                  new NonRootRankActor(
                    rankRepository,
                    { (s: String) =>
                      q.offer(s).unsafeRunSync()
                    }
                  )
                )
              )
            }
          } yield q

          val getUniqueId = IO
            .fromFuture(
              IO(
                actorSystem
                  .actorSelection(
                    s"akka://cifar-0@${config.rootAddress}:${config.rootPort}/user/uid"
                  )
                  .resolveOne(600 seconds)
              )
            )
            .flatMap { actorRef =>
              IO.fromFuture(IO(actorRef.ask("ask-id")(60 seconds)))
                .map(_.asInstanceOf[String])
                .map(s => NcclUniqueId(s))
            }

          def joinClique(id: NcclUniqueId) = IO.blocking(
            STen.ncclInitComm(
              nRanks = config.nranks,
              myRank = config.rank,
              myDevice = config.gpu,
              ncclUniqueId = id
            )
          )

          def trainEpoch(comm: NcclComm): IO[Double] =
            DistributedDataParallel.oneEpoch(
              model = model,
              stepOptimizerFn = None,
              trainBatches = trainBatches(),
              logger = Some(scribe.Logger()),
              accumulateGradientOverNBatches = 1,
              ncclComm = comm,
              rootRank = 0,
              device = device,
              forwardOnly = false
            )

          def loop(q: Queue[IO, String], comm: NcclComm): IO[Unit] =
            q.take.flatMap { command =>
              command match {
                case "train" => trainEpoch(comm) *> loop(q, comm)
                case "stop"  => IO.unit
              }
            }

          for {
            id <- getUniqueId
            q <- getCommandQueue
            comm <- joinClique(id)
            r <- loop(q, comm)
          } yield r

        }

        program.unsafeRunSync()
        actorSystem.terminate()

        ()
      }
    case _ =>
    // arguments are bad, error message will have been displayed
  }
  scribe.info("END")
}
