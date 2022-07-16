package lamp.example.cifar

import java.io.File

import lamp.CudaDevice
import lamp.data.BatchStream
import lamp.nn.SupervisedModel
import lamp.nn.AdamW
import lamp.nn.simple
import lamp.nn.LossFunctions

import lamp.SinglePrecision
import lamp.Scope
import lamp.STen
import cats.effect.unsafe.implicits.global
import lamp.data.distributed
import scala.concurrent.duration._
import com.typesafe.config.ConfigFactory

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

object Train {
  def main(args: Array[String]) : Unit = {
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

          val validationBatches = () =>
            BatchStream
              .minibatchesFromFull(
                config.testBatchSize,
                true,
                testFullbatch,
                testTarget,
                rng
              )
              .withoutEmptyBatches
              .everyNth(n = config.nranks, offset = config.rank)

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

            val comm = new lamp.distributed.akka.AkkaCommunicationServer(
              actorSystem
            )
            distributed.driveDistributedTraining(
              nranks = config.nranks,
              gpu = config.gpu,
              controlCommunication = comm,
              model = model,
              optimizerFactory = AdamW.factory(
                weightDecay = simple(0.00),
                learningRate = simple(config.learningRate)
              ),
              trainBatches = trainBatches,
              validationBatches = validationBatches,
              maxEpochs = config.epochs
            )

          } else {

            val comm = new lamp.distributed.akka.AkkaCommunicationClient(
              actorSystem,
              config.rootAddress,
              config.rootPort,
              "cifar-0",
              600 seconds
            )

            distributed.followDistributedTraining(
              rank = config.rank,
              nranks = config.nranks,
              gpu = config.gpu,
              controlCommunication = comm,
              model = model,
              trainBatches = trainBatches,
              validationBatches = validationBatches
            )

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
}
