package lamp.experiment.recursivelm.model
import lamp.data._
import lamp._
import lamp.autograd.const
object DataLoader {
  def minibatchesFromDocuments(
      minibatchSize: Int,
      numBatches: Int,
      corpus: STen,
      blockLength: Int,
      recursionLength: Int,
  ) = {
    val N = corpus.shape(0)
    val rng = scala.util.Random
    def makeNonEmptyBatch(device: Device) = {
      BatchStream.scopeInResource.map { implicit scope =>
        val (tokens, targets) = Scope { implicit scope =>
          val minibatches = 0 until minibatchSize map { _ =>
            
           val start0 = rng.nextLong(N - blockLength * recursionLength * 2 - 1)
            val starts = {
              0 until recursionLength map { i =>
                  start0 + i*blockLength 
              }
              
            }

            starts.map { start =>
              val token = corpus
                .slice(0, start.toLong, start.toLong + blockLength, 1L)
                .castToLong
              val target = corpus
                .slice(
                  0,
                  start.toLong + 1L,
                  start.toLong + blockLength + 1L,
                  1L
                )
                .castToLong
              (token, target)
            }
          }

          val minibatchTokens = minibatches
            .map(_.map(_._1))
            .transpose
            .map(st => STen.stack(st, dim = 0))
          val minibatchTargets = minibatches
            .map(_.map(_._2))
            .transpose
            .map(st => STen.stack(st, dim = 0))

          assert(minibatchTargets.size == recursionLength)
          assert(minibatchTokens.size == recursionLength)

          device.withOtherStreamThenSync(synchronizeBefore = false) {

            (
              minibatchTokens.map(device.to),
              minibatchTargets.map(device.to)
            )
          }
        }

        val batch = LossInput(
          tokens.map(const) zip targets
        )

        val fakeTarget =
          STen.zeros(List(minibatchSize), device.options(SinglePrecision))

        NonEmptyBatch((batch, fakeTarget))
      }

    }

    BatchStream.fromFunction(numBatches, makeNonEmptyBatch)

  }
}
