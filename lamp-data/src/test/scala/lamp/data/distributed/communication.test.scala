package lamp.data.distributed

import org.scalatest.funsuite.AnyFunSuite
import cats.effect.std.Queue
import cats.effect.IO
import cats.implicits._
import lamp.data.distributed.DistributedCommunication._
import lamp.NcclUniqueId
import scala.concurrent.duration._
import cats.effect.unsafe.implicits.global

object CommunicationTest {
  def makeTest(
      nranks: Int,
      comms: (
          DistributedCommunicationRoot,
          Seq[DistributedCommunicationNonRoot]
      )
  ) = {
    def waitForPeers(comm: DistributedCommunicationRoot): IO[Unit] =
      for {
        n <- comm.peers()
        r <-
          if (n < nranks - 1) IO.sleep(1 second) *> waitForPeers(comm)
          else IO.unit
      } yield r
    def root(comm: DistributedCommunicationRoot) = {
      for {
        _ <- comm.onUniqueIdReady(NcclUniqueId("fake"))
        _ <- waitForPeers(comm)
        _ <- comm.broadcast(Train)
        _ <- comm.broadcast(Valid)
        _ <- comm.broadcast(Train)
        _ <- comm.broadcast(Stop)
      } yield ()
    }
    def nonroots(comms: Seq[DistributedCommunicationNonRoot]) = {

      def receiveLoop(
          q: Queue[IO, DistributedCommunication.Command],
          acc: Vector[DistributedCommunication.Command]
      ): IO[Vector[DistributedCommunication.Command]] = {
        for {
          c <- q.take
          r <- c match {
            case Stop  => IO.pure(acc :+ Stop)
            case Train => receiveLoop(q, acc :+ Train)
            case Valid => receiveLoop(q, acc :+ Valid)
          }
        } yield r
      }

      comms.map { comm =>
        for {
          q <- Queue.bounded[IO, DistributedCommunication.Command](1)
          _ <- comm.join(q)
          r <- receiveLoop(q, Vector.empty)
        } yield r
      }
    }

    for {
      fiber <- nonroots(comms._2).parSequence.start
      _ <- root(comms._1)
      outcome <- fiber.joinWithNever
    } yield {
      scribe.info(outcome.toString)
      assert(
        outcome == Vector(
          Vector(Train, Valid, Train, Stop),
          Vector(Train, Valid, Train, Stop)
        )
      )
      outcome
    }
  }
}

class CommunicationSuiteSuite extends AnyFunSuite {

  test("local comm") {

    val nranks = 3
    LocalCommunication
      .make(nranks)
      .flatMap(c => CommunicationTest.makeTest(nranks, c))
      .unsafeRunSync()

  }
}
