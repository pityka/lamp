package lamp.example.lm
import lamp._
import lamp.nn._
import lamp.data.IdentityCodecFactory
import lamp.autograd.Autograd

object Model {

  val vocabularySize = 256
  val contextLength = 384

  val codecFactory = IdentityCodecFactory

  def allocateModel(
      device: Device,
      gradientCheckpointing: Boolean,
      mixedPrecision: Boolean
  )(implicit
      scope: Scope
  ) = {
    val tensorOptions =
      device.options(if (mixedPrecision) Bf16Precision else SinglePrecision)
    val embeddingDim = 768
    val layers = 12
    val numHeads = 12
    val net = lamp.nn.languagemodel.LanguageModelLoss.apply(
      maxLength = contextLength,
      vocabularySize = vocabularySize,
      numBlocks = layers,
      embeddingDim = embeddingDim,
      attentionHiddenPerHeadDim = embeddingDim / numHeads,
      attentionNumHeads = numHeads,
      encoderMlpHiddenDim = embeddingDim * 4,
      dropout = 0d,
      padToken = -1000L,
      tOpt = tensorOptions,
      linearized = false
    )
    scribe.info(
      f"Allocated model on $device . embedding=$embeddingDim layers=$layers num-heads=$numHeads num-param=${net.learnableParameters}%,d gradient-checkpointing=$gradientCheckpointing"
    )

    SupervisedModel(
      net,
      LossFunctions.Identity,
      // printMemoryAllocations = true,
      cacheStrategy =
        if (gradientCheckpointing) Autograd.CacheStrategy.SelectivelyCache
        else Autograd.CacheStrategy.AlwaysCache
    )
  }

}
