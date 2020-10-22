package lamp.example.cifar

import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import org.saddle._
import lamp.TensorHelpers
import lamp.util.syntax
import aten.ATen
import cats.effect.Resource
import cats.effect.IO
import lamp.CudaDevice
import lamp.CPU
import lamp.data.BatchStream
import lamp.data.IOLoops
import lamp.nn.SupervisedModel
import lamp.data.TrainingCallback
import lamp.data.ValidationCallback
import aten.Tensor
import lamp.nn.AdamW
import lamp.nn.simple
import lamp.nn.LossFunctions
import lamp.data.BufferedImageHelper
import lamp.nn.LearningRateSchedule

import lamp.data.Reader
import lamp.DoublePrecision
import lamp.FloatingPointPrecision
import lamp.SinglePrecision
import lamp.Scope

object Cifar {
  def loadImageFile(
      file: File,
      numImages: Int,
      precision: FloatingPointPrecision
  ) = {
    val inputStream = new FileInputStream(file)
    val channel = inputStream.getChannel()
    val tensors = 0 until numImages map { _ =>
      val bb = ByteBuffer
        .allocate(3074)
      lamp.data.Reader.readFully(bb, channel)
      bb.get
      val label2 = bb.get

      val to = ByteBuffer.allocate(3072)
      while (to.hasRemaining() && bb.hasRemaining()) {
        to.asInstanceOf[ByteBuffer].put(bb.get)
      }
      val vec = Vec(to.array).map(b => (b & 0xff).toDouble)
      val red = Mat(vec.slice(0, 1024).copy).reshape(32, 32)
      val green = Mat(vec.slice(1024, 2 * 1024).copy).reshape(32, 32)
      val blue = Mat(vec.slice(2 * 1024, 3 * 1024).copy).reshape(32, 32)
      val tensor = TensorHelpers.fromMatList(
        List(red, green, blue),
        device = CPU,
        precision = precision
      )
      (label2, tensor)
    }
    val labels =
      TensorHelpers.fromLongVec(tensors.map(_._1.toLong).toVec, false)
    val fullBatch = {
      val tmp = ATen.cat(tensors.map(_._2).toArray, 0)
      val f = ATen._unsafe_view(tmp, Array(-1, 3, 32, 32))
      tmp.release()
      f
    }

    tensors.foreach(_._2.release())
    (labels, fullBatch)

  }
}

case class CliConfig(
    trainData: String = "",
    testData: String = "",
    labels: String = "",
    cuda: Boolean = false,
    trainBatchSize: Int = 32,
    testBatchSize: Int = 32,
    epochs: Int = 10,
    learningRate: Double = 0.001,
    dropout: Double = 0.0,
    network: String = "lenet",
    checkpointSave: Option[String] = None,
    checkpointLoad: Option[String] = None,
    singlePrecision: Boolean = false
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
      opt[Unit]("gpu").action((_, c) => c.copy(cuda = true)),
      opt[Unit]("single").action((_, c) => c.copy(singlePrecision = true)),
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
        val device = if (config.cuda) CudaDevice(0) else CPU
        val precision =
          if (config.singlePrecision) SinglePrecision else DoublePrecision
        val tensorOptions = device.options(precision)
        val model = {
          val numClasses = 100
          val classWeights = ATen.ones(Array(numClasses), tensorOptions)
          val net =
            // if (config.network == "lenet")
            //   Cnn.lenet(numClasses, dropOut = config.dropout, tensorOptions)
            Cnn.resnet(numClasses, config.dropout, tensorOptions)

          val loadedNet = config.checkpointLoad match {
            case None => net
            case Some(file) =>
              Reader
                .loadFromFile(net, new File(file), device)
                .unsafeRunSync()
                .right
                .get
          }
          scribe.info("Learnable parametes: " + loadedNet.learnableParameters)
          scribe.info("parameters: " + loadedNet.parameters.mkString("\n"))
          SupervisedModel(
            loadedNet,
            LossFunctions.NLL(numClasses, classWeights)
          )
        }

        val (trainTarget, trainFullbatch) =
          Cifar.loadImageFile(new File(config.trainData), 50000, precision)
        val (testTarget, testFullbatch) =
          Cifar.loadImageFile(new File(config.testData), 10000, precision)
        Resource
          .fromAutoCloseable(IO {
            scala.io.Source.fromFile(config.labels)
          })
          .use(src => IO { src.getLines.toVector })
          .unsafeRunSync

        scribe.info(
          s"Loaded full batch data. Train shape: ${trainFullbatch.shape}"
        )
        val rng = org.saddle.spire.random.rng.Cmwc5.apply
        val trainEpochs = () =>
          BatchStream.minibatchesFromFull(
            config.trainBatchSize,
            true,
            trainFullbatch,
            trainTarget,
            device,
            rng
          )
        val testEpochs = () =>
          BatchStream.minibatchesFromFull(
            config.testBatchSize,
            true,
            testFullbatch,
            testTarget,
            device,
            rng
          )

        val optimizer = AdamW.factory(
          weightDecay = simple(0.00),
          learningRate = simple(config.learningRate),
          scheduler = LearningRateSchedule.noop
        )

        IOLoops
          .epochs(
            model = model,
            optimizerFactory = optimizer,
            trainBatchesOverEpoch = trainEpochs,
            validationBatchesOverEpoch = Some(testEpochs),
            epochs = config.epochs,
            trainingCallback = TrainingCallback.noop,
            validationCallback =
              ValidationCallback.logAccuracy(scribe.Logger("validation")),
            checkpointFile = config.checkpointSave.map(s => new File(s)),
            minimumCheckpointFile =
              config.checkpointSave.map(s => new File(s + ".min")),
            logger = Some(scribe.Logger("training"))
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
  def showImage(t: Tensor): javax.swing.JFrame = {
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
