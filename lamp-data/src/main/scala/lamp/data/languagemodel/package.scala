package lamp.data

import lamp._
import lamp.data.BatchStream.scopeInResource
import lamp.autograd.const
import lamp.nn.languagemodel.LossInput
import lamp.nn.languagemodel.LanguageModelInput
import lamp.nn.languagemodel.LanguageModelModule
import cats.effect.kernel.Resource
import lamp.nn.languagemodel.LanguageModelOutputNonVariable
import lamp.nn.languagemodel.LanguageModelOutput
import lamp.nn.GenericFun
import cats.effect.IO

package object languagemodel {

  def autoregressiveInference(
      model: LanguageModelModule,
      modelBlockSize: Int,
      prefix: Array[Char],
      length: Int,
      temperature: Double
  )(scope: Scope): IO[Array[Short]] = {
    assert(prefix.length < modelBlockSize)
    assert(temperature > 0d)

    def makeInput(prefix: Array[Short])(implicit scope: Scope) = {
      val tokens =
        STen
          .fromLongArray(
            prefix.map(_.toLong).take(modelBlockSize)
          )
          .unsqueeze(0)

      val positions = STen.fromLongArray(Array(tokens.shape(1) - 1)).unsqueeze(0)

      val maxLength = {
        val single = STen.arange_l(0, tokens.shape(1), 1).unsqueeze(0)
        single.repeat(List(1, 1))
      }

      LanguageModelInput(
        tokens = const(tokens),
        maxLength = Some(maxLength),
        positions = Some(positions)
      )
    }

    def makeBatch(prefix: Array[Char]) =
      BatchStream.single(scopeInResource.map { implicit scope =>
        NonEmptyBatch(makeInput(prefix))
      })

    def single(
        prefix: Array[Char]
    )(implicit scope: Scope): IO[LanguageModelOutputNonVariable] =
      IOLoops
        .runBatchStream(
          makeBatch(prefix),
          buffers = Resource.unit,
          model = lamp.nn.sequence(
            model,
            GenericFun[LanguageModelOutput, LanguageModelOutputNonVariable](_ =>
              _.toSTen
            )
          )
        )
        .map(_.head)

    def loop(n: Int, acc: Array[Char])(scope: Scope): IO[Array[Char]] =
      if (n == 0) IO.pure(acc)
      else
        Scope
          .bracket(scope) { implicit scope =>
            val prefix = acc.takeRight(modelBlockSize)
            single(prefix).map { output =>
              val probs = output.languageModelScores.exp.view(1, -1)

              val sample = STen.multinomial(
                (probs / temperature).logSoftMax(1).exp,
                1,
                false
              )
              assert(sample.numel == 1)
              val next = sample.toLongArray.head.toChar
              next
            }
          }
          .flatMap(next => loop(n - 1, acc :+ next)(scope))

    loop(length, prefix)(scope)

  }

  def autoregressiveMinibatchesFromCorpus(
      minibatchSize: Int,
      numBatches: Int,
      corpus: Array[Char],
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
