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

object LanguageModelInput {
  // implicit val movable: Movable[LanguageModelInput] =
  //   Movable.by(v => (v.tokens, v.memory, v.positions))
}

/** Language model input and target for loss calculation
  *
  * @param input
  * @param languageModelTarget
  *   batch x sequence
  */
case class LossInput(
    tokensAndTarget: Seq[(Constant, STen)]
) {
  // assert(
  //   languageModelTarget.shape == positions
  //     .map(_.shape)
  //     .getOrElse(tokens.shape)
  // )
}

object LossInput {
  // implicit val movable: Movable[LossInput] =
  //   Movable.by(v => (v.input, v.languageModelTarget))
}

/** Module with the language model and a loss
  *
  * Main trainig entry point of the language model
  */
case class LanguageModelLoss(
    languageModel: LanguageModelModule,
    loss: LossFunction,
    memoryWidth: Int
) extends GenericModule[LossInput, Variable] {
  def state = languageModel.state
  def forward[S: Sc](x: LossInput): Variable = {

    val (_, likelihoods, targets) = x.tokensAndTarget.foldLeft(
      (Option.empty[Variable], Seq.empty[Variable], Seq.empty[STen])
    ) { case ((memory, prevLikelihoods, prevTargets), (tokens, target)) =>
      val input = LanguageModelInput(tokens, memory, None)
      val output = languageModel.forward(input)

      val likeLihoods =
        output.languageModelLogits
          .slice(dim = 1, start = memoryWidth, end = tokens.shape(1), 1)
          .logSoftMax(dim = 2)
          .flatten(0, 1)
      val targets = target
        .slice(dim = 1, start = memoryWidth, end = target.shape(1), 1)
        .view(-1L)
      val memory2 =
        output.encoded.slice(dim = 1, start = 0, end = memoryWidth, step = 1)
      (Some(memory2), prevLikelihoods :+ likeLihoods, prevTargets :+ targets)
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
      memoryWidth: Int,
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
    memoryWidth = memoryWidth,
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
    val embedded0 =
      tokenEmbedding.forward(x.tokens) + positionEmbedding.forward(pos)

    val embedded =
      if (x.memory.isEmpty) embedded0
      else
        x.memory.get.cat(
          embedded0
            .slice(dim = 1, x.memory.get.shape(1), embedded0.shape(1), 1),
          dim = 1
        )
    val encoded = finalNorm(encoder.forward((embedded, None)))

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
      gptOrder = true,
      causalMask = true
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
