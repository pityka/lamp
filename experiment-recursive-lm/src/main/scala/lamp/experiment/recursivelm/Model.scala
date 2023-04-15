package lamp.experiment.recursivelm
import lamp._
import lamp.nn._
import lamp.data.bytesegmentencoding.ByteSegmentCodecFactory

object Model {

  val vocabularySize = 128
  val contextLength = 64
  val recursionLength = 4
  val memoryWidth = 32

  val codecFactory = ByteSegmentCodecFactory(
    vocabularyMin = 10,
    vocabularyMax = (vocabularySize - 1).toChar,
    maxMergedSegmentLength = 1,
    unknownToken = 0.toChar,
    unknownByte = '?'.toByte
  )

  def allocateModel(device: Device)(implicit
      scope: Scope
  ) = {
    val tensorOptions = device.options(SinglePrecision)
    val embeddingDim = 768
    val layers = 3
    val numHeads = 12
    val net = lamp.experiment.recursivelm.model.LanguageModelLoss.apply(
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
      linearized = false,
      memoryWidth = memoryWidth
    )
    scribe.info(
      f"Allocated model on $device . embedding=$embeddingDim layers=$layers num-heads=$numHeads num-param=${net.learnableParameters}%,d"
    )

    // scribe.info(s"List of parameters: \n${net.parameters
    //   .map(v => v._2.getClass -> v._1.value.numel)
    //   .mkString("\n")}")
    SupervisedModel(net, LossFunctions.Identity)
  }

}
