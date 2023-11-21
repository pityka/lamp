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

/** Data loader and inference utilities for the language model module in
  * lamp.nn.langaugemodel
  */
package object languagemodel {

  /** Recursive single next token inference of LanguageModelModule
    *
    * @param model
    * @param modelBlockSize
    *   also known as context length or maximum length of model
    * @param prefix
    *   The inference starts from this prefix sequence
    * @param length
    *   Length of inference. Each inferred token is appended to the prefix and
    *   fed back to the model after truncating to modelBlockSize from the right.
    * @param temperature
    *   Sampling temperature. 1.0 means no change from model output, <1 less
    *   random, >1 more random.
    * @return
    */
  def autoregressiveInference(
      model: LanguageModelModule,
      modelBlockSize: Int,
      prefix: Array[Char],
      length: Int,
      temperature: Double
  )(scope: Scope): IO[Array[Char]] = {
    assert(temperature > 0d)
    val device = model.tokenEmbedding.weights.value.device
    def makeInput(prefix: Array[Char])(implicit scope: Scope) = {
      val tokens =
        STen
          .fromLongArray(
            prefix.map(_.toLong)
          )
          .unsqueeze(0)

      val positions =
        STen.fromLongArray(Array(tokens.shape(1) - 1)).unsqueeze(0)

      val maxLength = {
        val single = STen.arange_l(1, tokens.shape(1) + 1, 1).unsqueeze(0)
        single.repeat(List(1, 1))
      }

      LanguageModelInput(
        tokens = const(device.to(tokens)),
        maxLength = Some(device.to(maxLength)),
        positions = Some(device.to(positions))
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
              val probs = (output.languageModelLogits / temperature)
                .logSoftMax(2)
                .exp
                .view(1, -1)

              val sample = STen.multinomial(
                probs,
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

  /** Creates random minibatches of fixed size from an in memory corpus. The
    * attention mask is set up for autoregressive (left-to-right / causal)
    * attention.
    *
    * @param minibatchSize
    *   Number of sequences in the minibatch
    * @param numBatches
    *   Number of minibatches to generate
    * @param corpus
    *   Tokens of corpus 1D int32 tensor
    * @param blockLength
    *   Length of sequences, also known as context length
    * @return
    */
  def autoregressiveMinibatchesFromCorpus(
      minibatchSize: Int,
      numBatches: Int,
      corpus: STen,
      blockLength: Int,
      createMaxLength: Boolean = true
  ) = {
    def makeNonEmptyBatch(device: Device) = {
      scopeInResource.evalMap { implicit scope =>
        IO.interruptible {
          val corpusLength = corpus.shape(0)
          val (tokens, targets, maxLength) = Scope { implicit scope =>
            val starts = STen
              .randint(
                low = 0,
                high = corpusLength - blockLength - 1,
                size = List(minibatchSize),
                STenOptions.l
              )
              .toLongArray
              .toVector
              .map(_.toLong)

            val tokens = STen.stack(
              starts.map(i =>
                corpus.slice(0L, i.toLong, i + blockLength, 1L).castToLong
              ),
              dim = 0
            )
            val targets = STen.stack(
              starts.map(i =>
                corpus.slice(0L, i + 1L, i + 1 + blockLength, 1L).castToLong
              ),
              dim = 0
            )

            val maxLength = if (createMaxLength) {
              val single = STen.arange_l(1, blockLength + 1, 1).unsqueeze(0)
              Some(single.repeat(List(minibatchSize, 1)))
            } else None
            device.withOtherStream(
              synchronizeBefore = true,
              synchronizeAfter = true
            ) {
              // println(s"TX ENTER ${device.asInstanceOf[lamp.CudaDevice].getCurrentStream} ${Thread.currentThread().getName}")
              val k = (
                device.to(tokens),
                device.to(targets),
                maxLength.map(device.to)
              )
// println(s"TX EXIT ${device.asInstanceOf[lamp.CudaDevice].getCurrentStream} ${Thread.currentThread().getName}")
              k
            }
          }

          val batch = LossInput(
            input = LanguageModelInput(
              tokens = const(tokens),
              maxLength = maxLength,
              positions = None
            ),
            languageModelTarget = targets
          )

          val fakeTarget = STen.zeros(List(minibatchSize), tokens.options)

          NonEmptyBatch((batch, fakeTarget))
        }
      }
    }

    BatchStream.fromFunction(numBatches, makeNonEmptyBatch)

  }

}
