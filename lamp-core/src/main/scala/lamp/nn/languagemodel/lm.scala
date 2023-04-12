/** Neural network components of a language model
  *
  * This can be used to set up a GPT like autoregressive neural network or BERT
  * like masked language model. lamp.nn.bert also implements BERT.
  */
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

/** Input to language model
  *
  * @param tokens
  *   batch x sequence, type long
  * @param maxLength
  *   batch x sequence OR batch, see maskedSoftmax. Used to define masking of
  *   the attention matrix. Use cases:
  *   - Left-to-right (causal) attention with uniform sequence length. In this
  *     case use a batch x sequence 2D matrix with arange(0,sequence) in each
  *     row.
  *   - Variable length sequences with bidirectional attention. In this case use
  *     a 1D [batch] vector with the real length of each sequence (rest are
  *     padded).
  *   - If empty the attention matrix is not masked
  * @param positions
  *   batch x sequence, type long in [0,sequence], selects positions. Final LM
  *   logits are computed on the selected positions. If empty then selects all
  *   positions.
  */
case class LanguageModelInput(
    tokens: Constant,
    maxLength: Option[STen],
    positions: Option[STen]
)

object LanguageModelInput {
  implicit val movable: Movable[LanguageModelInput] =
    Movable.by(v => (v.tokens, v.maxLength, v.positions))
}

/** Language model input and target for loss calculation
  *
  * @param input
  * @param languageModelTarget
  *   batch x sequence
  */
case class LossInput(
    input: LanguageModelInput,
    languageModelTarget: STen
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

/** Module with the language model and a loss
  *
  * Main trainig entry point of the language model
  */
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
  */
case class LanguageModelOutput(
    encoded: Variable,
    languageModelLogits: Variable
) {
  def toSTen =
    LanguageModelOutputNonVariable(encoded.value, languageModelLogits.value)
}

/* Same as LanguageModelOutput but holds raw tensors, not variables */
case class LanguageModelOutputNonVariable(
    encoded: STen,
    languageModelLogits: STen
)

object LanguageModelOutputNonVariable {
  implicit val movable: Movable[LanguageModelOutputNonVariable] =
    Movable.by(v => (v.encoded, v.languageModelLogits))
}

/** Transformer based language model module
  *
  * Initial embedding is the sum of token and position embedding. Token
  * embedding is a learned embedding. Position embedding is also a learned
  * embedding (not sinusoidal etc).
  *
  * Initial embeddings are fed into layers of transformer blocks. Attention
  * masking is governed by the input similarly as described in chapter 11.3.2.1
  * in d2l v1.0.0-beta0.
  *
  * Selected sequence positions in the output of the transformer chain are
  * linearly mapped back into the desired vocabulary size.
  */
case class LanguageModelModule(
    tokenEmbedding: Embedding,
    positionEmbedding: Embedding,
    encoder: TransformerEncoder,
    finalNorm: LayerNorm
) extends GenericModule[LanguageModelInput, LanguageModelOutput] {
  def state =
    tokenEmbedding.state ++ positionEmbedding.state ++ encoder.state ++ finalNorm.state // ++ lmHead.state

  def forward[S: Sc](x: LanguageModelInput): LanguageModelOutput = {

    val pos = const(
      STen.arange_l(0, x.tokens.shape(1), 1, x.tokens.options).unsqueeze(0)
    )
    val embedded =
      tokenEmbedding.forward(x.tokens) + positionEmbedding.forward(pos)
    val encoded = finalNorm(encoder.forward((embedded, x.maxLength)))

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

    def mm1(a: Variable, b: Variable) = {
      val shape = a.shape
      a.view(List(-1, shape.last)).mm(b).view(shape.dropRight(1) :+ -1L)
    }

    val logits =
      mm1(encoderOutputAtPredictionPositions, tokenEmbedding.weights.t)
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
      linearized = linearized,
      gptOrder = true
    ),
    finalNorm = LayerNorm(List(embeddingDim.toLong), tOpt)
  )

  implicit val trainingMode: TrainingMode[LanguageModelModule] = TrainingMode
    .make[LanguageModelModule](
      m =>
        m.copy(
          tokenEmbedding = m.tokenEmbedding.asEval,
          positionEmbedding = m.positionEmbedding.asEval,
          encoder = m.encoder.asEval,
          finalNorm = m.finalNorm.asEval
        ),
      m =>
        m.copy(
          tokenEmbedding = m.tokenEmbedding.asTraining,
          positionEmbedding = m.positionEmbedding.asTraining,
          encoder = m.encoder.asTraining,
          finalNorm = m.finalNorm.asTraining
        )
    )
  implicit val load: Load[LanguageModelModule] =
    Load.compose(
      _.tokenEmbedding,
      _.positionEmbedding,
      _.encoder,
      _.finalNorm
    )

}
