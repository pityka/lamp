package lamp.experiment.recursivelm.model

import lamp._
import lamp.nn._
import lamp.autograd.const
import cats.effect._
import lamp.data._
import BatchStream.scopeInResource

object Sampling {
  def autoregressiveInference(
      model: LanguageModelModule,
      modelBlockSize: Int,
      modelMemoryWidth: Int,
      prefix: Array[Char],
      length: Int,
      temperature: Double
  )(scope: Scope): IO[Array[Char]] = {
    assert(temperature > 0d)
    assert(prefix.size > modelMemoryWidth)
    val device = model.tokenEmbedding.weights.value.device
    def makeInput(memory: STen, prefix: Array[Char])(implicit scope: Scope) = {
      val tokens =
        STen
          .fromLongArray(
            prefix.map(_.toLong)
          )
          .unsqueeze(0)

      val positions =
        STen.fromLongArray(Array(tokens.shape(1) - 1)).unsqueeze(0)

      LanguageModelInput(
        tokens = const(device.to(tokens)),
        memory = const(memory),
        positions = Some(device.to(positions)),
        memoryWidth = modelMemoryWidth
      )
    }

    def makeBatch(memory: STen, prefix: Array[Char]) =
      BatchStream.single(scopeInResource.map { implicit scope =>
        NonEmptyBatch(makeInput(memory, prefix))
      })

    def single(
        memory: STen,
        prefix: Array[Char]
    )(implicit scope: Scope): IO[LanguageModelOutputNonVariable] =
      IOLoops
        .runBatchStream(
          makeBatch(memory, prefix),
          buffers = Resource.unit,
          model = lamp.nn.sequence(
            model,
            GenericFun[LanguageModelOutput, LanguageModelOutputNonVariable](_ =>
              _.toSTen
            )
          )
        )
        .map(_.head)

    def loop(n: Int, acc: Array[Char], memory: STen)(
        scope: Scope
    ): IO[Array[Char]] =
      if (n == 0) IO.pure(acc)
      else
        Scope
          .bracket(scope) { implicit scope =>
            val prefix = acc.takeRight(modelBlockSize)
            single(memory, prefix).map { output =>
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
              val memory = output.encoded.slice(
                dim = 1,
                start = 0,
                end = modelMemoryWidth,
                step = 1
              )
              (next, memory)
            }
          }
          .flatMap { case (next, memory) =>
            loop(n - 1, acc :+ next, memory)(scope)
          }

    val initMemory = STen.zeros(
      List(1, modelMemoryWidth, model.tokenEmbedding.weights.sizes(1)),
      device.options(SinglePrecision)(scope)
    )(scope)
    loop(length, prefix, initMemory)(scope)

  }
}
