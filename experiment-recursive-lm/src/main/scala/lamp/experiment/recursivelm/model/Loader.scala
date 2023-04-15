package lamp.experiment.recursivelm.model
import lamp.data._
import lamp._
import lamp.autograd.const
import scala.collection.immutable
object DataLoader {
  def minibatchesFromDocuments(
      minibatchSize: Int,
      numBatches: Int,
      documents: Array[STen],
      blockLength: Int
  ) = {
    val N = documents.length
    val rng = scala.util.Random
    def makeNonEmptyBatch(device: Device) = {
      BatchStream.scopeInResource.map { implicit scope =>
        val (tokens, targets) = Scope { implicit scope =>
          val minibatches = 0 until minibatchSize map {_ =>
          val doc = documents(rng.nextInt(N))
          val docLength = doc.shape(0).toInt

          def makeIndex(n: Int, acc: List[Int]): List[Int] =
            n match {
              case 0 => acc.reverse
              case _ =>
                acc match {
                  case head :: _ =>
                    if (head < docLength - blockLength - 1)
                      makeIndex(
                        n - 1,
                        (rng.nextInt(
                          docLength - blockLength - 1 - head
                        ) + head) :: acc
                      )
                    else acc.reverse
                  case immutable.Nil =>
                    makeIndex(
                      n - 1,
                      rng.nextInt(
                        docLength - blockLength - 1
                      ) :: Nil
                    )
                }
            }
          val starts = makeIndex(3,Nil)

          starts.map{ start =>
            val token = doc.slice(0,start.toLong,start.toLong+blockLength,1L).castToLong  
            val target = doc.slice(0,start.toLong+1L,start.toLong+blockLength+1L,1L).castToLong  
            (token,target)
          }
        }

        val minibatchTokens = minibatches.map(_.map(_._1)).transpose.map(st => STen.stack(st,dim=1))
        val minibatchTargets = minibatches.map(_.map(_._2)).transpose.map(st => STen.stack(st,dim=1))

          

          device.withOtherStreamThenSync(synchronizeBefore = false) {

            (
              minibatchTokens.map(device.to),
              minibatchTargets.map(device.to),
            )
          }
        }


        val batch = LossInput(
         tokens.map(const) zip targets
        )

        val fakeTarget = STen.zeros(List(minibatchSize), device.options(SinglePrecision))

        NonEmptyBatch((batch, fakeTarget))
      }

    }

    BatchStream.fromFunction(numBatches, makeNonEmptyBatch)

  }
}
