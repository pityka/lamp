package lamp.example.gan

import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import org.saddle._
import lamp.TensorHelpers
import aten.ATen
import cats.effect.IO
import lamp.CudaDevice
import lamp.CPU
import lamp.data.BatchStream
import lamp.autograd.const
import lamp.nn.AdamW
import lamp.nn.simple
import lamp.data.BufferedImageHelper

import lamp.FloatingPointPrecision
import lamp.SinglePrecision
import lamp.Scope
import lamp.STen
import lamp.nn.LossFunctions.BCEWithLogits
import lamp.STenOptions
import lamp.data.Writer
import javax.imageio.ImageIO

object GAN {
  def loadImageFile(
      file: File,
      numImages: Int,
      precision: FloatingPointPrecision
  )(implicit scope: Scope) = {
    Scope { implicit scope =>
      val inputStream = new FileInputStream(file)
      val channel = inputStream.getChannel()
      val tensors = (0 until numImages)
        .map { _ =>
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
          (label1, label2, tensor)
        }
      val labels =
        TensorHelpers.fromLongVec(tensors.map(_._2.toLong).toVec, false)
      val fullBatch = {
        val tmp = ATen.cat(tensors.map(_._3).toArray, 0)
        val f = ATen._unsafe_view(tmp, Array(-1, 3, 32, 32))
        tmp.release()
        f
      }

      val t1 = STen.owned(fullBatch)

      tensors.foreach(_._3.release())
      (STen.owned(labels), t1)
    }

  }
}

case class CliConfig(
    trainData: String = "",
    cuda: Boolean = false,
    trainBatchSize: Int = 512,
    epochs: Int = 10
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
      opt[Unit]("gpu").action((_, c) => c.copy(cuda = true)),
      opt[Int]("batch-train").action((x, c) => c.copy(trainBatchSize = x)),
      opt[Int]("epochs").action((x, c) => c.copy(epochs = x))
    )

  }

  val latentDim = 100
  def makeNoise(instances: Int, tensorOptions: STenOptions)(implicit
      scope: Scope
  ) =
    const(STen.randn(List(instances, latentDim, 1, 1), tensorOptions))

  OParser.parse(parser1, args, CliConfig()) match {
    case Some(config) =>
      scribe.info(s"Config: $config")
      Scope.root { implicit scope =>
        val device =
          if (config.cuda && aten.Tensor.cudnnAvailable()) CudaDevice(0)
          else CPU
        val precision = SinglePrecision
        val tensorOptions = device.options(precision)
        val discriminator = Cnn.discriminator(3, tensorOptions)

        val generator = Cnn.generator(latentDim, tensorOptions)

        val loss = BCEWithLogits(STen.scalarDouble(1d, tensorOptions))

        val (trainTarget, trainFullbatch) =
          GAN.loadImageFile(
            new File(config.trainData),
            50000,
            precision
          )

        scribe.info(
          s"Loaded full batch data. Train shape: ${trainFullbatch.shape}"
        )
        val rng = org.saddle.spire.random.rng.Cmwc5.apply()
        val trainEpochs = () =>
          BatchStream.minibatchesFromFull(
            config.trainBatchSize,
            false,
            trainFullbatch,
            trainTarget,
            device,
            rng
          )
        val lr = 0.0005

        val optimizerDiscriminator = AdamW.factory(
          weightDecay = simple(0.00),
          learningRate = simple(lr)
          // beta1 = simple(0.5)
        )(discriminator.parameters.map(v => (v._1.value, v._2)))
        val optimizerGenerator = AdamW.factory(
          weightDecay = simple(0.00),
          learningRate = simple(lr)
          // beta1 = simple(0.5)
        )(generator.parameters.map(v => (v._1.value, v._2)))

        // val updateWindow = AWTWindow.showImage(
        //   generator.forward(makeNoise(1, tensorOptions)).value.select(0, 0)
        // )

        0 until config.epochs foreach { epoch =>
          Scope { implicit scope =>
            val batches = trainEpochs()

            val totalGLoss = STen.scalarDouble(0d, tensorOptions)
            val totalDLoss = STen.scalarDouble(0d, tensorOptions)
            var instances = 0L

            batches
              .foldLeft(()) { case (_, (feature, _)) =>
                Scope.inResource.use { implicit scope =>
                  IO {
                    val batchSize = feature.shape(0)
                    val noise =
                      makeNoise(batchSize.toInt, tensorOptions)
                    val ones = STen.ones(List(batchSize), tensorOptions)
                    val zeros = STen.zeros(List(batchSize), tensorOptions)
                    val discriminatorOutput =
                      discriminator.forward(feature)
                    val generatorOutput = generator.forward(noise)
                    val discriminatorOutputOnGenerated =
                      discriminator.forward(generatorOutput.detached)
                    val discriminatorLoss =
                      (loss(discriminatorOutput, ones)._1 + loss(
                        discriminatorOutputOnGenerated,
                        zeros
                      )._1) * 0.5
                    optimizerDiscriminator.step(
                      discriminator.gradients(discriminatorLoss),
                      1d
                    )

                    val discriminatorOutputOnGenerated2 =
                      discriminator.forward(generatorOutput)
                    val generatorLoss =
                      loss(discriminatorOutputOnGenerated2, ones)._1

                    optimizerGenerator.step(
                      generator.gradients(generatorLoss),
                      1d
                    )

                    STen.addOut(
                      totalDLoss,
                      totalDLoss,
                      discriminatorLoss.value,
                      batchSize.toDouble
                    )
                    STen.addOut(
                      totalGLoss,
                      totalGLoss,
                      generatorLoss.value,
                      batchSize.toDouble
                    )
                    instances += batchSize

                  }
                }
              }
              .unsafeRunSync()

            println(
              s"D-loss: ${totalDLoss.toMat.raw(0) / instances}, G-loss: ${totalGLoss.toMat
                .raw(0) / instances}"
            )

            // updateWindow(
            //   generator.forward(makeNoise(1, tensorOptions)).value.select(0, 0)
            // )
            println(
              AWTWindow.save(
                generator
                  .forward(makeNoise(1, tensorOptions))
                  .value
                  .select(0, 0),
                epoch
              )
            )
          }
        }

        Writer.writeCheckpoint(new File("generator.data"), generator)

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
  def save(t: STen, i: Int) = {

    val image = BufferedImageHelper.fromDoubleTensor(t)
    ImageIO.write(image, "png", new File(s"gen_$i.png"));
    new File(s"gen_$i.png").getAbsoluteFile()
  }
  def showImage(t: STen) = {
    println(t)
    import javax.swing._
    import java.awt.{Graphics}
    var image = BufferedImageHelper.fromDoubleTensor(t)
    println(image)
    val frame = new JFrame("");
    frame.setDefaultCloseOperation(javax.swing.WindowConstants.HIDE_ON_CLOSE);
    frame
      .getContentPane()
      .add(
        new JPanel {
          override def paintComponent(g: Graphics) = {
            super.paintComponent(g)
            val g2 = g.asInstanceOf[Graphics2D]
            val at = new AffineTransform();
            g2.drawImage(image, at, null);

          }
        },
        java.awt.BorderLayout.CENTER
      );

    frame.setSize(200, 200)
    frame.setVisible(true);
    val update = (t: STen) => {
      image = BufferedImageHelper.fromDoubleTensor(t)
      frame.repaint()
    }
    update
  }

}
