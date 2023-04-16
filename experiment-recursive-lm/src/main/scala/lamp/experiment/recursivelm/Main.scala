package lamp.experiment.recursivelm

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
            _ <-
                Train.train(config, corpora._1, corpora._2)(scope)
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
