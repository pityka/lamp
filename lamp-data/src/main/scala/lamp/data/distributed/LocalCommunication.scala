package lamp.data.distributed

import cats.effect._
import cats.effect.std.Queue
import cats.implicits._
import lamp.NcclUniqueId

object LocalCommunication {
  class LocalIOCommunicationServer(
      ranks: Ref[IO, List[Queue[IO, DistributedCommunication.Command]]],
      storedUid: Deferred[IO, NcclUniqueId]
  ) extends DistributedCommunicationRoot {

    def onUniqueIdReady(uid: NcclUniqueId): IO[Unit] =
      storedUid.complete(uid).map(_ => ())

    def peers(): IO[Int] = ranks.get.map(_.size)
    def broadcast(
        command: DistributedCommunication.Command
    ): IO[Unit] =
      for {
        qs <- ranks.get
        _ <- qs.map(_.offer(command)).sequence
      } yield ()

  }
  class LocalIOCommunicationClient(
      ranks: Ref[IO, List[Queue[IO, DistributedCommunication.Command]]],
      storedUid: Deferred[IO, NcclUniqueId]
  ) extends DistributedCommunicationNonRoot {

    def join(
        q: Queue[IO, DistributedCommunication.Command]
    ): IO[NcclUniqueId] = for {
      _ <- ranks.update(old => q :: old)
      id <- storedUid.get
    } yield id

  }
  def make(nranks: Int) = {
    for {
      ranks <- Ref.of[IO, List[Queue[IO, DistributedCommunication.Command]]](
        Nil
      )
      savedUid <- Deferred[IO, NcclUniqueId]
    } yield {
      val server = new LocalIOCommunicationServer(ranks, savedUid)
      val clients = 1 until nranks map (_ =>
        new LocalIOCommunicationClient(ranks, savedUid)
      )
      (server, clients)
    }
  }
}
