package lamp.example.cifar

import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import org.saddle._
import lamp.autograd.TensorHelpers
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
import lamp.syntax
import lamp.data.BufferedImageHelper
import lamp.nn.SGDW
import lamp.nn.LearningRateSchedule
import org.saddle.scalar.ScalarTagDouble
import lamp.data.Writer
import java.io.FileOutputStream
import lamp.data.Reader
import lamp.DoublePrecision
import lamp.FloatingPointPrecision

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
      val label1 = bb.get
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
    // {
    //   val first = ATen.select(fullBatch, 0, 1)
    //   val firstRed = ATen.select(first, 0, 1)
    //   println(firstRed.toMat)
    // }
    // {
    //   val normalized = fullBatch.normalized
    //   val first = ATen.select(normalized, 0, 1)
    //   val firstRed = ATen.select(first, 0, 1)
    //   println(firstRed.toMat)
    // }
    // ???
    // println(tensors.map(_._1.toLong).apply(1))
    // AWTWindow.showImage(first)
    // ???

    tensors.foreach(_._2.release())
    (labels, fullBatch)

  }
}

case class CliConfig(
    trainData: String = "",
    testData: String = "",
    labels: String = "",
    cuda: Boolean = false,
    trainBatchSize: Int = 256,
    testBatchSize: Int = 256,
    epochs: Int = 1000,
    learningRate: Double = 0.001,
    dropout: Double = 0.0,
    network: String = "lenet",
    checkpointSave: Option[String] = None,
    checkpointLoad: Option[String] = None
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
      opt[Unit]("gpu").action((x, c) => c.copy(cuda = true)),
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
      val device = if (config.cuda) CudaDevice(0) else CPU
      val precision = DoublePrecision
      val tensorOptions = device.options(precision)
      val model: SupervisedModel[Unit] = {
        val numClasses = 100
        val classWeights = ATen.ones(Array(numClasses), tensorOptions)
        val net =
          if (config.network == "lenet")
            Cnn.lenet(numClasses, dropOut = config.dropout, tensorOptions)
          else Cnn.resnet(32, 32, numClasses, config.dropout, tensorOptions)

        // val loadedNet = config.checkpointLoad match {
        //   case None => net
        //   case Some(file) =>
        //     Reader.loadFromFile(net, new File(file))
        // }
        scribe.info("Learnable parametes: " + net.learnableParameters)
        scribe.info("parameters: " + net.parameters.mkString("\n"))
        SupervisedModel(net, (), LossFunctions.NLL(numClasses, classWeights))
      }

      val (trainTarget, trainFullbatch) =
        Cifar.loadImageFile(new File(config.trainData), 50000, precision)
      val (testTarget, testFullbatch) =
        Cifar.loadImageFile(new File(config.testData), 10000, precision)
      val textLabels =
        Resource
          .fromAutoCloseable(IO {
            scala.io.Source.fromFile(config.labels)
          })
          .use(src => IO { src.getLines.toVector })
          .unsafeRunSync

      scribe.info(
        s"Loaded full batch data. Train shape: ${trainFullbatch.shape}"
      )

      val trainEpochs = () =>
        BatchStream.minibatchesFromFull(
          config.trainBatchSize,
          true,
          trainFullbatch,
          trainTarget,
          device
        )
      val testEpochs = () =>
        BatchStream.minibatchesFromFull(
          config.testBatchSize,
          true,
          testFullbatch,
          testTarget,
          device
        )

      val optimizer = AdamW.factory(
        weightDecay = simple(0.00),
        learningRate = simple(config.learningRate),
        scheduler = LearningRateSchedule.cyclicSchedule(10d, 200L)
      )

      val trainedModel = IOLoops
        .epochs(
          model = model,
          optimizerFactory = optimizer,
          trainBatchesOverEpoch = trainEpochs,
          validationBatchesOverEpoch = testEpochs,
          epochs = config.epochs,
          trainingCallback = TrainingCallback.noop,
          validationCallback = ValidationCallback.logAccuracy,
          checkpointFile = config.checkpointSave.map(s => new File(s)),
          minimumCheckpointFile = None,
          checkpointFrequency = 10,
          logger = Some(scribe.Logger("training"))
        )
        .unsafeRunSync()

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
    import java.awt.{Graphics, RenderingHints}
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
            val bounds = getBounds()
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
