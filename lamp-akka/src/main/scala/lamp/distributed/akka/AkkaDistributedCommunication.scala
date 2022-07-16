package lamp.distributed.akka
import lamp.data.distributed._
import lamp._
import _root_.akka.actor._
import _root_.akka.pattern.ask
import cats.effect.IO
import scala.concurrent.duration._
import DistributedCommunication._
import cats.effect.std.Queue

class AkkaCommunicationServer(
    actorSystem: ActorSystem
) extends DistributedCommunicationRoot {
  val ranksServer = actorSystem.actorOf(Props(new RankRepository), "ranks")

  private val getRanks = IO
    .fromFuture(IO(ranksServer.ask("get")(5 seconds)))
    .map(_.asInstanceOf[List[ActorRef]])
  def onUniqueIdReady(uid: NcclUniqueId): IO[Unit] = IO {
    actorSystem.actorOf(Props(new UniqueIdServer(uid)), "uid")
    ()
  }

  def peers(): IO[Int] = getRanks.map(_.size)
  def broadcast(
      command: DistributedCommunication.Command
  ): IO[Unit] = {
    val encodedCommand = command match {
      case Stop  => "stop"
      case Train => "train"
      case Valid => "valid"
    }
    for {
      ranks <- getRanks
      _ <- IO(
        if (ranks.isEmpty) {
          scribe.info(s"Should broadcast $command but no peers around")
          throw new RuntimeException(
            "broadcast must be called after onNcclCliqueReady completed"
          )
        } else {
          scribe.info(s"Broadcasting $command to ${ranks.size} peers")
          ranks.foreach(_ ! encodedCommand)
        }
      )
    } yield ()

  }

}
class AkkaCommunicationClient(
    actorSystem: ActorSystem,
    rootAddress: String,
    rootPort: Int,
    rootActorSystemName: String,
    timeoutWaitingForServer: FiniteDuration
) extends DistributedCommunicationNonRoot {

  private def retry[A, B](io: IO[Either[Throwable, B]], n: Int): IO[B] = {
    io.flatMap {
      case Right(b) => IO.pure(b)
      case Left(ActorNotFound(_)) if n > 0 =>
        IO(scribe.info("Wait..")) *> IO.sleep(5 seconds) *> IO(
          scribe.info("Retry")
        ) *> retry(io, n - 1)
      case Left(e) => IO.raiseError(e)
    }
  }

  private val getUniqueId: IO[NcclUniqueId] = {

    retry(
      IO
        .fromFuture(
          IO(
            actorSystem
              .actorSelection(
                s"akka://${rootActorSystemName}@${rootAddress}:${rootPort}/user/uid"
              )
              .resolveOne(timeoutWaitingForServer)
          )
        )
        .attempt,
      60
    )
      .flatMap { actorRef =>
        IO.fromFuture(IO(actorRef.ask("ask-id")(60 seconds)))
          .map(_.asInstanceOf[String])
          .map(s => NcclUniqueId(s))
          .flatTap(s => IO(scribe.info(s"Got unique id $s")))
      }
  }

  def join(q: Queue[IO, Command]): IO[NcclUniqueId] =
    for {
      id <- getUniqueId
      rankRepository <-
        IO
          .fromFuture(
            IO(
              actorSystem
                .actorSelection(
                  s"akka://${rootActorSystemName}@${rootAddress}:${rootPort}/user/ranks"
                )
                .resolveOne(60 seconds)
            )
          )
      _ <- IO(scribe.info("Got actorref to rank repository"))
      _ <- IO {
        actorSystem.actorOf(
          Props(
            new NonRootRankActor(
              rankRepository,
              { (s: String) =>
                val decoded = s match {
                  case "stop"  => DistributedCommunication.Stop
                  case "train" => DistributedCommunication.Train
                  case "valid" => DistributedCommunication.Valid
                }
                scribe.info(s"Received $decoded")
                import cats.effect.unsafe.implicits.global
                q.offer(decoded).unsafeRunSync()
              }
            )
          )
        )
      }
    } yield id
}
