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
import lamp.SinglePrecision
import lamp.Scope
import lamp.STen
import lamp.onnx.VariableInfo
import lamp.data.{EndStream, EmptyBatch, NonEmptyBatch}
import cats.effect.unsafe.implicits.global

object Cifar {
  def loadImageFile(
      file: File,
      numImages: Int,
      precision: FloatingPointPrecision,
      pin: Boolean
  )(implicit scope: Scope) = {
    Scope { implicit scope =>
      val all = STen
        .fromFile(
          file.getAbsolutePath(),
          offset = 0,
          length = numImages * 3074,
          scalarTypeByte = 0,
          pin = pin
        )
        .view(numImages.toLong, -1)
      val label2 = all.select(1, 1).castToLong
      val images0 = all.slice(1, 2, 3074, 1).view(-1, 3, 32, 32)
      val images = precision match {
        case SinglePrecision => images0.castToFloat
        case DoublePrecision => images0.castToDouble
      }
      println(images)
      (label2, images)
    }

  }
}

case class CliConfig(
    trainData: String = "",
    testData: String = "",
    labels: String = "",
    gpus: List[Int] = Nil,
    trainBatchSize: Int = 32,
    testBatchSize: Int = 32,
    epochs: Int = 10,
    learningRate: Double = 0.001,
    dropout: Double = 0.0,
    network: String = "lenet",
    checkpointSave: Option[String] = None,
    checkpointLoad: Option[String] = None,
    singlePrecision: Boolean = false,
    pinnedAllocator: Boolean = false
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
      opt[String]("label-data")
        .action((x, c) => c.copy(labels = x))
        .text("path to cifar100 fine label file")
        .required(),
      opt[String]("gpus").action((v, c) =>
        c.copy(gpus = v.split(",").toList.filterNot(_.isEmpty).map(_.toInt))
      ),
      opt[Unit]("single").action((_, c) => c.copy(singlePrecision = true)),
      opt[Unit]("pinned").action((_, c) => c.copy(pinnedAllocator = true)),
      opt[Int]("batch-train").action((x, c) => c.copy(trainBatchSize = x)),
      opt[Int]("batch-test").action((x, c) => c.copy(testBatchSize = x)),
      opt[Int]("epochs").action((x, c) => c.copy(epochs = x)),
      opt[Double]("learning-rate").action((x, c) => c.copy(learningRate = x)),
      opt[Double]("dropout").action((x, c) => c.copy(dropout = x)),
      opt[String]("network").action((x, c) => c.copy(network = x)),
      opt[String]("checkpoint-save").action((x, c) =>
        c.copy(checkpointSave = Some(x))
      ),
      opt[String]("checkpoint-load").action((x, c) =>
        c.copy(checkpointLoad = Some(x))
      )
    )

  }

  OParser.parse(parser1, args, CliConfig()) match {
    case Some(config) =>
      scribe.info(s"Config: $config")
      Scope.root { implicit scope =>
        val devices =
          if (config.gpus.isEmpty) List(CPU) else config.gpus.map(CudaDevice)
        val device = devices.head
        val extraDevices = devices.drop(1)
        if (config.pinnedAllocator) {
          aten.Tensor.setPinnedMemoryAllocator()
        }
        val precision =
          if (config.singlePrecision) SinglePrecision else DoublePrecision
        val tensorOptions = device.options(precision)
        val model = {
          val numClasses = 100
          val classWeights = STen.ones(List(numClasses), tensorOptions)
          val net =
            // if (config.network == "lenet")
            //   Cnn.lenet(numClasses, dropOut = config.dropout, tensorOptions)
            Cnn.resnet(numClasses, config.dropout, tensorOptions)

          config.checkpointLoad match {
            case None =>
            case Some(file) =>
              Reader
                .loadFromFile(net, new File(file), device, false)
                .unsafeRunSync()

          }
          scribe.info("Learnable parametes: " + net.learnableParameters)
          SupervisedModel(
            net,
            LossFunctions.NLL(numClasses, classWeights),
            printMemoryAllocations = false
          )
        }

        val models = extraDevices.map { device =>
          val numClasses = 100
          val classWeights =
            STen.ones(List(numClasses), device.to(tensorOptions))
          val net =
            // if (config.network == "lenet")
            //   Cnn.lenet(numClasses, dropOut = config.dropout, tensorOptions)
            Cnn.resnet(numClasses, config.dropout, device.to(tensorOptions))

          scribe.info("Learnable parametes: " + net.learnableParameters)
          SupervisedModel(
            net,
            LossFunctions.NLL(numClasses, classWeights)
          )
        }

        val (trainTarget, trainFullbatch) =
          Cifar.loadImageFile(
            new File(config.trainData),
            50000,
            precision,
            config.pinnedAllocator
          )
        val (testTarget, testFullbatch) =
          Cifar.loadImageFile(
            new File(config.testData),
            10000,
            precision,
            config.pinnedAllocator
          )
        Resource
          .fromAutoCloseable(IO {
            scala.io.Source.fromFile(config.labels)
          })
          .use(src => IO { src.getLines().toVector })
          .unsafeRunSync()

        scribe.info(
          s"Loaded full batch data. Train shape: ${trainFullbatch.shape}"
        )
        val rng = org.saddle.spire.random.rng.Cmwc5.apply()
        val trainEpochs = () =>
          BatchStream.minibatchesFromFull(
            config.trainBatchSize,
            true,
            trainFullbatch,
            trainTarget,
            rng
          )
        val testEpochs = () =>
          BatchStream.minibatchesFromFull(
            config.testBatchSize,
            true,
            testFullbatch,
            testTarget,
            rng
          )

        val optimizer = AdamW.factory(
          weightDecay = simple(0.00),
          learningRate = simple(config.learningRate)
        )

        val (_, trained, _) = IOLoops
          .epochs(
            model = model,
            optimizerFactory = optimizer,
            trainBatchesOverEpoch = trainEpochs,
            validationBatchesOverEpoch = Some(testEpochs),
            epochs = config.epochs,
            logger = Some(scribe.Logger("training")),
            dataParallelModels = models,
            printOptimizerAllocations = false
          )
          .unsafeRunSync()

        testEpochs()
          .nextBatch(device, 0)
          .flatMap(
            _._2
              .use {
                case NonEmptyBatch(batch) =>
                  IO {
                    val output = trained.module.forward(batch._1)
                    val file = new java.io.File("cifar10.lamp.example.onnx")
                    lamp.onnx.serializeToFile(
                      file,
                      output,
                      domain = "lamp.example.cifar"
                    ) {
                      case x if x == output =>
                        VariableInfo(
                          variable = output,
                          name = "output",
                          input = false,
                          docString = "log probabilities"
                        )
                      case x if x == batch._1 =>
                        VariableInfo(
                          variable = batch._1,
                          name = "input",
                          input = true,
                          docString = "Nx3xHxW"
                        )
                    }
                    println(
                      "Model exported to ONNX into file" + file.getAbsolutePath
                    )
                  }
                case EndStream | EmptyBatch =>
                  IO { println("First batch is empty or stream ended") }

              }
          )
          .unsafeRunSync()

        ()
      }
    case _ =>
    // arguments are bad, error message will have been displayed
  }
  scribe.info("END")
}
object AWTWindow {
  import java.awt.Graphics2D
  import java.awt.geom.AffineTransform
  def showImage(t: STen): javax.swing.JFrame = {
    import javax.swing._
    import java.awt.{Graphics}
    val image = BufferedImageHelper.fromDoubleTensor(t)
    val frame = new JFrame("");
    frame.setDefaultCloseOperation(javax.swing.WindowConstants.HIDE_ON_CLOSE);
    frame
      .getContentPane()
      .add(
        new JComponent {
          override def paintComponent(g: Graphics) = {
            super.paintComponent(g)
            val g2 = g.asInstanceOf[Graphics2D]
            val at = new AffineTransform();
            g2.drawImage(image, at, null);

          }
        },
        java.awt.BorderLayout.CENTER
      );

    frame.pack();
    frame.setVisible(true);
    frame
  }

}
