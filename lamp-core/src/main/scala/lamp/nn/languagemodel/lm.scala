package lamp.nn.languagemodel

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
    // batch x sequence, type long
    tokens: Constant,
    // batch x sequence OR batch, see maskedSoftmax
    maxLength: Option[STen],
    // batch x sequence, type long in [0,sequence], selects positions
    positions: Option[STen]
)

object LanguageModelInput {
  implicit val movable: Movable[LanguageModelInput] =
    Movable.by(v => (v.tokens, v.maxLength, v.positions))
}

case class LossInput(
    input: LanguageModelInput,
    languageModelTarget: STen // batch x sequence
) {
  assert(
    languageModelTarget.shape == input.positions
      .map(_.shape)
      .getOrElse(input.tokens.shape)
  )
}

object LossInput {
  implicit val movable: Movable[LossInput] =
    Movable.by(v => (v.input, v.languageModelTarget))
}

case class LanguageModelLoss(
    languageModel: LanguageModelModule,
    loss: LossFunction
) extends GenericModule[LossInput, Variable] {
  def state = languageModel.state
  def forward[S: Sc](x: LossInput): Variable = {
    val output = languageModel.forward(x.input)
    val (l1, _) =
      loss
        .apply(
          output.languageModelLogits.logSoftMax(dim = 2).flatten(0, 1),
          x.languageModelTarget.view(-1L)
        )

    l1
  }
}

object LanguageModelLoss {

  /** Allocate module
    *
    * @param maxLength
    *   Total sequence length including cls, sep and potential pad tokens
    * @param vocabularySize
    *   Total vocabulary size including cls, sep, pad, mask tokens
    * @param numBlocks
    *   Number of transformer blocks
    * @param embeddingDim
    *   Width of the initial embedding dimension, as well as the output width of
    *   the feed forward network in each transformer block
    * @param attentionHiddenPerHeadDim
    * @param attentionNumHeads
    * @param encoderMlpHiddenDim
    *   Hidden dimension within transformer blocks
    * @param dropout
    * @param padToken
    *   pad will be ignored
    * @param tOpt
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
      linearized: Boolean
  ): LanguageModelLoss = LanguageModelLoss(
    languageModel = LanguageModelModule(
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
      ignore = padToken // not sure this is needed
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
  *   - encoded: float tensor of size (batch, sequence length, embedding
  *     dimension ) holds per token embeddings
  *   - languageModelScores: float tensor of size (batch, sequence length,
  *     vocabulary size) holds per token log probability distributions (from
  *     logSoftMax)
  *
  * @param encoded
  * @param languageModelScores
  * @param wholeSentenceBinaryClassifierScore
  */
case class LanguageModelOutput(
    encoded: Variable,
    languageModelLogits: Variable
) {
  def toSTen =
    LanguageModelOutputNonVariable(encoded.value, languageModelLogits.value)
}
case class LanguageModelOutputNonVariable(
    encoded: STen,
    languageModelLogits: STen
)

object LanguageModelOutputNonVariable {
  implicit val movable: Movable[LanguageModelOutputNonVariable] =
    Movable.by(v => (v.encoded, v.languageModelLogits))
}

case class LanguageModelModule(
    tokenEmbedding: Embedding,
    positionEmbedding: Embedding,
    encoder: TransformerEncoder,
    lmHead: Linear
) extends GenericModule[LanguageModelInput, LanguageModelOutput] {
  def state =
    tokenEmbedding.state ++ positionEmbedding.state ++ encoder.state ++ lmHead.state

  def forward[S: Sc](x: LanguageModelInput): LanguageModelOutput = {

    val pos = const(
      STen.arange_l(0, x.tokens.shape(1), 1, x.tokens.options).unsqueeze(0)
    )
    val embedded =
      tokenEmbedding.forward(x.tokens) + positionEmbedding.forward(pos)
    val encoded = encoder.forward((embedded, x.maxLength))

    val encoderOutputAtPredictionPositions =
      x.positions.fold(encoded)(positions =>
        encoded
          .view(List(-1, encoded.shape(2)))
          .indexSelect(dim = 0, index = const(positions.view(-1)))
          .view(
            List(
              encoded.shape(0),
              positions.shape(1),
              encoded.shape(2)
            )
          )
      )

    val logits =
      lmHead.forward(encoderOutputAtPredictionPositions)
    LanguageModelOutput(
      encoded = encoded,
      languageModelLogits = logits
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
      linearized: Boolean
  ): LanguageModelModule = LanguageModelModule(
    tokenEmbedding = Embedding.apply(
      classes = vocabularySize,
      dimensions = embeddingDim,
      tOpt = tOpt
    ),
    positionEmbedding = Embedding.apply(
      classes = maxLength,
      dimensions = embeddingDim,
      tOpt = tOpt
    ),
    encoder = TransformerEncoder(
      numBlocks = numBlocks,
      in = embeddingDim,
      attentionHiddenPerHeadDim = attentionHiddenPerHeadDim,
      attentionNumHeads = attentionNumHeads,
      mlpHiddenDim = encoderMlpHiddenDim,
      dropout = dropout,
      tOpt = tOpt,
      linearized = linearized
    ),
    lmHead = Linear(
      in = embeddingDim,
      out = vocabularySize,
      tOpt = tOpt
    )
  )

  implicit val trainingMode: TrainingMode[LanguageModelModule] = TrainingMode
    .make[LanguageModelModule](
      m =>
        m.copy(
          tokenEmbedding = m.tokenEmbedding.asEval,
          positionEmbedding = m.positionEmbedding.asEval,
          encoder = m.encoder.asEval,
          lmHead = m.lmHead.asEval
        ),
      m =>
        m.copy(
          tokenEmbedding = m.tokenEmbedding.asTraining,
          positionEmbedding = m.positionEmbedding.asTraining,
          encoder = m.encoder.asTraining,
          lmHead = m.lmHead.asTraining
        )
    )
  implicit val load: Load[LanguageModelModule] =
    Load.compose(_.tokenEmbedding, _.positionEmbedding, _.encoder, _.lmHead)

}
