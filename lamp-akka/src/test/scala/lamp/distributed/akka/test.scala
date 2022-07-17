package lamp.distributed.akka

import org.scalatest.funsuite.AnyFunSuite
import scala.concurrent.duration._

import cats.effect.unsafe.implicits.global
import com.typesafe.config.ConfigFactory

class CommunicationSuiteSuite extends AnyFunSuite {

  test("local comm") {
    val nranks = 3
    val as = akka.actor.ActorSystem(
      name = s"name",
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
      canonical.hostname = "localhost"
      canonical.port = 28888
    }
  }
}
                """
        )
      )
    )
    val server = new AkkaCommunicationServer(as)
    val clients = 0 until 2 map (_ =>
      new AkkaCommunicationClient(as, "localhost", 28888, "name", 600 seconds)
    )
    lamp.data.distributed.CommunicationTest
      .makeTest(nranks, (server, clients))
      .unsafeRunSync()

    as.terminate()

  }
}
