package lamp.experiment.recursivelm.model

import lamp.autograd.Variable
import lamp.Sc
import lamp.nn._

import lamp.STen
import lamp.autograd.Constant
import lamp.autograd.{const}
import lamp.STenOptions

import lamp.autograd.Mean
import lamp.Movable

case class LanguageModelInput(
    tokens: Constant,
    memory: Option[Variable],
    positions: Option[STen]
)

/** Language model input and target for loss calculation
  *
  * @param input
  * @param languageModelTarget
  *   batch x sequence
  */
case class LossInput(
    tokensAndTarget: Seq[(Constant, STen)]
)

object LossInput

/** Module with the language model and a loss
  *
  * Main trainig entry point of the language model
  */
case class LanguageModelLoss(
    languageModel: LanguageModelModule,
    loss: LossFunction,
    embeddingDim: Int
) extends GenericModule[LossInput, Variable] {
  def state = languageModel.state
  def forward[S: Sc](x: LossInput): Variable = {

    val (_, likelihoods, targets) = x.tokensAndTarget.foldLeft(
      (Option.empty[Variable], Seq.empty[Variable], Seq.empty[STen])
    ) { case ((memory, prevLikelihoods, prevTargets), (tokens, target)) =>
      val input = LanguageModelInput(tokens, memory, None)

      val output = languageModel.forward(input)

      val likelihoods =
        output.languageModelLogits
          .logSoftMax(dim = 2)
          .flatten(0, 1)
      val targets = target
        .reshape(-1L)
      val memory2 = output.memory
      (Some(memory2), prevLikelihoods :+ likelihoods, prevTargets :+ targets)
    }

    val (l1, _) =
      loss
        .apply(
          Variable.cat(likelihoods, dim = 0),
          STen.cat(targets, dim = 0)
        )

    l1
  }
}

object LanguageModelLoss {

  /** Allocate language model module with negative log likelihood loss
    *
    * @param maxLength
    *   Total sequence length including padding if used. Sometimes called block
    *   length or context length.
    * @param vocabularySize
    *   Total vocabulary size.
    * @param numBlocks
    *   Number of transformer blocks (layers).
    * @param embeddingDim
    *   Width of the initial embedding dimension, as well as the output width of
    *   each transformer block
    * @param attentionHiddenPerHeadDim
    *   Per head hidden dimension in the multihead attention
    * @param attentionNumHeads
    *   Number of attention heads in the multihead attention
    * @param encoderMlpHiddenDim
    *   Hidden dimension within transformer blocks
    * @param dropout
    * @param padToken
    *   This token is ignored during loss computation. Not used otherwise.
    * @param tOpt
    *   TensorOption to set device and data type
    * @param linearized
    *   Whether to use linearized self attention
    * @return
    */
  def apply[S: Sc](
      maxLength: Int,
      vocabularySize: Int,
      numBlocks: Int,
      embeddingDim: Int,
      attentionHiddenPerHeadDim: Int,
      attentionNumHeads: Int,
      encoderMlpHiddenDim: Int,
      dropout: Double,
      padToken: Long,
      tOpt: STenOptions,
      linearized: Boolean,
      memoryWidth: Int
  ): LanguageModelLoss = LanguageModelLoss(
    embeddingDim = embeddingDim,
    languageModel = LanguageModelModule(
      memoryWidth = memoryWidth,
      maxLength = maxLength,
      vocabularySize = vocabularySize,
      numBlocks = numBlocks,
      embeddingDim = embeddingDim,
      attentionNumHeads = attentionNumHeads,
      attentionHiddenPerHeadDim = attentionHiddenPerHeadDim,
      encoderMlpHiddenDim = encoderMlpHiddenDim,
      dropout = dropout,
      tOpt = tOpt,
      linearized = linearized
    ),
    loss = LossFunctions.NLL(
      numClasses = vocabularySize,
      classWeights = STen.ones(List(vocabularySize), tOpt),
      reduction = Mean,
      ignore = padToken
    )
  )

  implicit val trainingMode: TrainingMode[LanguageModelLoss] = TrainingMode
    .make[LanguageModelLoss](
      m =>
        m.copy(
          languageModel = m.languageModel.asEval
        ),
      m =>
        m.copy(
          languageModel = m.languageModel.asTraining
        )
    )
  implicit val load: Load[LanguageModelLoss] =
    Load.compose(_.languageModel)

}

/** Output of LM
  *
  * @param encoded
  *   encoded: float tensor of size (batch, sequence length, embedding
  *   dimension) holds per token embeddings
  * @param languageModelLogits
  *   float tensor of size (batch, sequence length, vocabulary size) holds per
  *   token logits. Use logSoftMax(dim=2) to get log probabilities.
  * @param memory
  *   float tnesor of size same as encoded
  */
case class LanguageModelOutput(
    encoded: Variable,
    languageModelLogits: Variable,
    memory: Variable
) {
  def toSTen =
    LanguageModelOutputNonVariable(
      encoded.value,
      languageModelLogits.value,
      memory.value
    )
}

/* Same as LanguageModelOutput but holds raw tensors, not variables */
case class LanguageModelOutputNonVariable(
    encoded: STen,
    languageModelLogits: STen,
    memory: STen
)

object LanguageModelOutputNonVariable {
  implicit val movable: Movable[LanguageModelOutputNonVariable] =
    Movable.by(v => (v.encoded, v.languageModelLogits, v.memory))
}

case class LanguageModelModule(
    tokenEmbedding: Embedding,
    positionEmbedding: Embedding,
    encoderDecoder: TransformerEncoder,
    // cross: Transformer,
    finalNorm: LayerNorm,
    memoryWidth: Int
    // memoryNorm: LayerNorm
) extends GenericModule[LanguageModelInput, LanguageModelOutput] {
  def state =
    tokenEmbedding.state ++
      positionEmbedding.state ++
      encoderDecoder.state ++
      // cross.state ++
      finalNorm.state
  // memoryNorm.state

  def forward[S: Sc](x: LanguageModelInput): LanguageModelOutput = {

    val nSeq = x.tokens.shape(1)
    val pos = const(
      STen.arange_l(0, nSeq + 2*memoryWidth, 1, x.tokens.options).unsqueeze(0)
    )
    val posEmbedding = positionEmbedding.forward(pos)
    val tokenEmbedded =
      tokenEmbedding.forward(x.tokens)
    val memory = x.memory.getOrElse(
      const(
        STen.zeros(
          List(x.tokens.shape(0), memoryWidth, posEmbedding.shape(2)),
          posEmbedding.options
        )
      )
    )

    val cat =
      Variable.cat(List(memory, tokenEmbedded, memory), dim = 1) + posEmbedding
    val encoded = finalNorm(encoderDecoder.forward((cat, None)))

    val newMemory = encoded.slice(dim = 1, nSeq + memoryWidth, nSeq + memoryWidth * 2, 1)

    val encodedTokens = encoded.slice(dim = 1, memoryWidth, nSeq +memoryWidth, 1)

    val encoderOutputAtPredictionPositions =
      x.positions.fold(encodedTokens)(positions =>
        encodedTokens
          .view(List(-1, encodedTokens.shape(2)))
          .indexSelect(dim = 0, index = const(positions.view(-1)))
          .view(
            List(
              encodedTokens.shape(0),
              positions.shape(1),
              encodedTokens.shape(2)
            )
          )
      )

    def mm1(a: Variable, b: Variable) = {
      val shape = a.shape
      a
        .reshape(List(-1, shape.last))
        .mm(b)
        .view(shape.dropRight(1) :+ -1L)
    }

    val logits =
      mm1(encoderOutputAtPredictionPositions, tokenEmbedding.weights.t)
    LanguageModelOutput(
      encoded = encodedTokens,
      languageModelLogits = logits,
      memory = newMemory
    )
  }

}

object LanguageModelModule {

  def apply[S: Sc](
      maxLength: Int,
      vocabularySize: Int,
      numBlocks: Int,
      embeddingDim: Int,
      attentionHiddenPerHeadDim: Int,
      attentionNumHeads: Int,
      encoderMlpHiddenDim: Int,
      dropout: Double,
      tOpt: STenOptions,
      linearized: Boolean,
      memoryWidth: Int
  ): LanguageModelModule = LanguageModelModule(
    memoryWidth = memoryWidth,
    tokenEmbedding = Embedding.apply(
      classes = vocabularySize,
      dimensions = embeddingDim,
      tOpt = tOpt
    ),
    positionEmbedding = Embedding.apply(
      classes = maxLength + 2*memoryWidth,
      dimensions = embeddingDim,
      tOpt = tOpt
    ),
    encoderDecoder = TransformerEncoder(
      numBlocks = numBlocks,
      in = embeddingDim,
      attentionHiddenPerHeadDim = attentionHiddenPerHeadDim,
      attentionNumHeads = attentionNumHeads,
      mlpHiddenDim = encoderMlpHiddenDim,
      dropout = dropout,
      tOpt = tOpt,
      linearized = linearized,
      gptOrder = true,
      causalMask = true
    ),
    // cross = Transformer(
    //   numBlocks = numBlocks,
    //   in = embeddingDim,
    //   attentionHiddenPerHeadDim = attentionHiddenPerHeadDim,
    //   attentionNumHeads = attentionNumHeads,
    //   mlpHiddenDim = encoderMlpHiddenDim,
    //   dropout = dropout,
    //   tOpt = tOpt,
    //   linearized = linearized,
    //   decoderDecoderCausalMask = false,
    //   encoderDecoderCausalMask = false
    // ),
    finalNorm = LayerNorm(List(embeddingDim.toLong), tOpt)
    // memoryNorm = LayerNorm(List(embeddingDim.toLong), tOpt)
  )

  implicit val trainingMode: TrainingMode[LanguageModelModule] = TrainingMode
    .make[LanguageModelModule](
      m =>
        m.copy(
          tokenEmbedding = m.tokenEmbedding.asEval,
          positionEmbedding = m.positionEmbedding.asEval,
          encoderDecoder = m.encoderDecoder.asEval,
          // cross = m.cross.asEval,
          finalNorm = m.finalNorm.asEval
          // memoryNorm = m.memoryNorm.asEval
        ),
      m =>
        m.copy(
          tokenEmbedding = m.tokenEmbedding.asTraining,
          positionEmbedding = m.positionEmbedding.asTraining,
          encoderDecoder = m.encoderDecoder.asTraining,
          // cross = m.cross.asTraining,
          finalNorm = m.finalNorm.asTraining
          // memoryNorm = m.memoryNorm.asTraining
        )
    )
  implicit val load: Load[LanguageModelModule] =
    Load.compose(
      _.tokenEmbedding,
      _.positionEmbedding,
      _.encoderDecoder,
      // _.cross,
      _.finalNorm
      // _.memoryNorm
    )

}
