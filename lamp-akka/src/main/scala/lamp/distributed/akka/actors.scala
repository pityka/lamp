package lamp.distributed.akka

import akka.actor._
import lamp.NcclUniqueId

class UniqueIdServer(uniqueIdBase64: NcclUniqueId) extends Actor {

  def receive: Receive = { case "ask-id" =>
    sender() ! uniqueIdBase64.base64
  }

}
class RankRepository extends Actor {
  val ranks = scala.collection.mutable.HashSet.empty[ActorRef]
  def receive: Receive = {
    case "rank" =>
      ranks.add(sender())
    case "get" => sender() ! ranks.toList
  }

}

class NonRootRankActor(rankRepository: ActorRef, callback: String => Unit)
    extends Actor {

  override def preStart(): Unit = {
    rankRepository ! "rank"
  }

  var listener = Option.empty[ActorRef]

  def receive: Receive = { case x: String =>
    callback(x)

  }

}
