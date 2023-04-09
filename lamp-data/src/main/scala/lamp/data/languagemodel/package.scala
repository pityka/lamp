package lamp.data

import lamp._
import lamp.data.BatchStream.scopeInResource
import lamp.autograd.const
import lamp.nn.languagemodel.LossInput
import lamp.nn.languagemodel.LanguageModelInput

package object languagemodel {

  def autoregressiveMinibatchesFromCorpus(
      minibatchSize: Int,
      numBatches: Int,
      corpus: Array[Short],
      blockLength: Int
  ) = {
    def makeNonEmptyBatch(device: Device) = {
      scopeInResource.map { implicit scope =>
        val (tokens, targets, maxLength) = Scope { implicit scope =>
          val starts = STen
            .randint(
              low = 0,
              high = corpus.length - blockLength - 1,
              size = List(minibatchSize),
              STenOptions.l
            )
            .toLongArray
            .toVector
            .map(_.toInt)

          val tokens = STen.stack(
            starts.map(i =>
              STen.fromLongArray(
                corpus.slice(i, i + blockLength).map(_.toLong)
              )
            ),
            dim = 0
          )
          val targets = STen.stack(
            starts.map(i =>
              STen.fromLongArray(
                corpus.slice(i + 1, i + 1 + blockLength).map(_.toLong)
              )
            ),
            dim = 0
          )

          val maxLength = {
            val single = STen.arange_l(0, blockLength, 1).unsqueeze(0)
            single.repeat(List(minibatchSize, 1))
          }

          device.withOtherStreamThenSync(synchronizeBefore = false) {

            (
              device.to(tokens),
              device.to(targets),
              device.to(maxLength)
            )
          }
        }
        // println("tokens")
        // println(tokens)
        // println("target")
        // println(targets)
        // println("validlen")
        // println(maxLength)
        // ???

        val batch = LossInput(
          input = LanguageModelInput(
            tokens = const(tokens),
            maxLength = Option(maxLength),
            positions = None
          ),
          languageModelTarget = targets
        )

        val fakeTarget = STen.zeros(List(minibatchSize), tokens.options)

        NonEmptyBatch((batch, fakeTarget))
      }

    }

    BatchStream.fromFunction(numBatches, makeNonEmptyBatch)

  }

}
