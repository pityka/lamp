package lamp.example.lm
import scopt.OParser
import cats.effect.IO
import cats.effect.ExitCode
import java.io.File

case class CliConfig(
    gpus: Seq[Int] = Nil,
    trainFile: String = "",
    validFile: String = "",
    fileMaxLength: Long = Long.MaxValue - 100,
    trainBatchSize: Int = 8,
    epochs: Int = 10000,
    beta2: Double = 0.95,
    learningRate: Double = 0.0001,
    weightDecay: Double = 0.0,
    samplingTemperature: Double = 1.0,
    dropout: Double = 0.0,
    numBatchesPerEpoch: Int = 100,
    checkpointSave: Option[String] = None,
    extend: Option[String] = None,
    extendLength: Int = 50,
    gradientAccumSteps: Int = 5,
    parallelism: Int = 64,
    // config for distributed training
    distributed: Boolean = false,
    gpu: Int = 0,
    rank: Int = 0,
    nranks: Int = 0,
    rootAddress: String = "",
    rootPort: Int = 28888,
    myAddress: String = "",
    myPort: Int = 28888
) {
  val bpeFile =
    checkpointSave.map(file => new File(file + ".bytesegmentencoding.json"))
}

object CliParser {

  def runCli(args: List[String])(b: CliConfig => IO[ExitCode]): IO[ExitCode] = {
    OParser.parse(parser1, args, CliConfig()) match {
      case Some(config) => b(config)
      case _            => IO.pure(ExitCode(1))
    }
  }

  private val builder = OParser.builder[CliConfig]
  val parser1 = {
    import builder._
    OParser.sequence(
      programName("languagemodel example"),
      head("trains an autoregressive language model"),
      opt[Seq[Int]]("gpus")
        .action((x, c) => c.copy(gpus = x))
        .text("list of gpus or empty for cpu"),
      opt[String]("train-file")
        .action((x, c) => c.copy(trainFile = x))
        .text("file containing ascii bytes"),
      opt[String]("valid-file")
        .action((x, c) => c.copy(validFile = x))
        .text("file containing ascii bytes"),
      opt[Int]("batch-size").action((x, c) => c.copy(trainBatchSize = x)),
      opt[Int]("parallelism").action((x, c) => c.copy(parallelism = x)),
      opt[String]("extend")
        .action((x, c) => c.copy(extend = Some(x)))
        .text("Turns on inference model. Extend this text in inference mode"),
      opt[Int]("extend-length")
        .action((x, c) => c.copy(extendLength = x))
        .text("extend this number of tkens in inference model"),
      opt[Long]("train-file-max-length").action((x, c) =>
        c.copy(fileMaxLength = x)
      ),
      opt[Int]("epochs").action((x, c) => c.copy(epochs = x)),
      opt[Int]("gradient-accum-steps").action((x, c) =>
        c.copy(gradientAccumSteps = x)
      ),
      opt[Int]("batches-per-epoch").action((x, c) =>
        c.copy(numBatchesPerEpoch = x)
      ),
      opt[Double]("learning-rate").action((x, c) => c.copy(learningRate = x)),
      opt[Double]("weight-decay").action((x, c) => c.copy(weightDecay = x)),
      opt[Double]("beta2").action((x, c) => c.copy(beta2 = x)),
      opt[Double]("dropout").action((x, c) => c.copy(dropout = x)),
      opt[Double]("sampling-temperature").action((x, c) =>
        c.copy(samplingTemperature = x)
      ),
      opt[String]("checkpoint").action((x, c) =>
        c.copy(checkpointSave = Some(x))
      ),
      opt[Unit]("distributed").action((_, c) => c.copy(distributed = true)),
      opt[Int]("rank").action((x, c) => c.copy(rank = x)),
      opt[Int]("nranks").action((x, c) => c.copy(nranks = x)),
      opt[String]("root-address").action((x, c) => c.copy(rootAddress = x)),
      opt[Int]("root-port").action((x, c) => c.copy(rootPort = x)),
      opt[String]("my-address").action((x, c) => c.copy(myAddress = x)),
      opt[Int]("my-port").action((x, c) => c.copy(myPort = x))
    )

  }
}
