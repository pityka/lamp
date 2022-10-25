package lamp.data.distributed

import cats.effect._
import cats.effect.std.Queue
import lamp.NcclUniqueId

object DistributedCommunication {
  sealed trait Command
  case object Train extends Command
  case object Valid extends Command
  case object Stop extends Command

}

trait DistributedCommunicationRoot {

  /** Framework will call this when the nccl unique id is ready
    *
    * Implementations should respond with this `uid` to each peer who try to
    * join this clique after this call.
    *
    * Returns a suspended side effect. Once the returned value is completed the
    * implementation is ready to accept peers to its clique.
    */
  def onUniqueIdReady(uid: NcclUniqueId): IO[Unit]

  /**
    * Returns the number of joined peers
    * 
    * Used for reporting and testing
    */
  def peers() : IO[Int]

  /** Broadcast command to all peers (non root ranks)
    *
    * @param state
    *   The value returned from `onNcclCliqueReady`
    * @param command
    * @return
    */
  def broadcast(
      command: DistributedCommunication.Command
  ): IO[Unit]
}
trait DistributedCommunicationNonRoot {

  /** Side effect which performs two actions:
    *   - joins the control communication clique and fetches the nccl unique id
    *     from the root rank
    *   - the server part of the control communiation clique may not be ready
    *     yet, in which case this client must try again
    *   - at the same time connects the given queue to the control communication
    *     clique's message channel. After this side effect is completed it is
    *     expected that commands are delivered to the give queue.
    *
    * @param q
    * @return
    */
  def join(
      q: Queue[IO, DistributedCommunication.Command]
  ): IO[NcclUniqueId]
}
