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
    // memoryWidth: Int
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
      linearized: Boolean
  ): LanguageModelLoss = LanguageModelLoss(
    embeddingDim = embeddingDim,
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
    encoder: Zip3Transformer,
    finalNorm: LayerNorm,
    memoryNorm: LayerNorm
) extends GenericModule[LanguageModelInput, LanguageModelOutput] {
  def state =
    tokenEmbedding.state ++ positionEmbedding.state ++ encoder.state ++ finalNorm.state ++ memoryNorm.state

  def forward[S: Sc](x: LanguageModelInput): LanguageModelOutput = {

    val pos = const(
      STen.arange_l(0, x.tokens.shape(1), 1, x.tokens.options).unsqueeze(0)
    )
    val embedded =
      tokenEmbedding.forward(x.tokens) + positionEmbedding.forward(pos)
    val memory = x.memory.getOrElse(embedded)

    val (encodedTokens, newMemory) = encoder.forward((embedded, memory))

    val encoded = finalNorm(
      encodedTokens
    )

    val normedMemroy = memoryNorm(newMemory)

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
      languageModelLogits = logits,
      memory = normedMemroy
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
    encoder = Zip3Transformer(
      numBlocks = numBlocks,
      in = embeddingDim,
      attentionHiddenPerHeadDim = attentionHiddenPerHeadDim,
      attentionNumHeads = attentionNumHeads,
      mlpHiddenDim = encoderMlpHiddenDim,
      dropout = dropout,
      tOpt = tOpt,
      linearized = linearized
    ),
    finalNorm = LayerNorm(List(embeddingDim.toLong), tOpt),
    memoryNorm = LayerNorm(List(embeddingDim.toLong), tOpt)
  )

  implicit val trainingMode: TrainingMode[LanguageModelModule] = TrainingMode
    .make[LanguageModelModule](
      m =>
        m.copy(
          tokenEmbedding = m.tokenEmbedding.asEval,
          positionEmbedding = m.positionEmbedding.asEval,
          encoder = m.encoder.asEval,
          finalNorm = m.finalNorm.asEval,
          memoryNorm = m.memoryNorm.asEval
        ),
      m =>
        m.copy(
          tokenEmbedding = m.tokenEmbedding.asTraining,
          positionEmbedding = m.positionEmbedding.asTraining,
          encoder = m.encoder.asTraining,
          finalNorm = m.finalNorm.asTraining,
          memoryNorm = m.memoryNorm.asTraining
        )
    )
  implicit val load: Load[LanguageModelModule] =
    Load.compose(
      _.tokenEmbedding,
      _.positionEmbedding,
      _.encoder,
      _.finalNorm,
      _.memoryNorm
    )

}

case class Zipped3(
    // encoder: TransformerEncoderBlock,
    decoder1: TransformerDecoderBlock,
    decoder2: TransformerDecoderBlock,
    norm: LayerNorm
) extends GenericModule[
      (Variable, Variable),
      (Variable, Variable)
    ] {

  def state = // encoder.state ++
    decoder1.state ++ decoder2.state ++ norm.state

  def forward[S: Sc](
      x: (Variable, Variable)
  ): (Variable, Variable) = {

    val (decoderInput, encoderInput) = x
    // val encoderOutput = encoder.forward((encoderInput, None))
    val decoder1Output =
      decoder1.forward((decoderInput, encoderInput, None))
    val decoder2Output =
      decoder2.forward((encoderInput, norm(decoder1Output), None))
    (decoder1Output, decoder2Output)

  }

}

object Zipped3 {

  /* Factory of a single transformer block */
  def apply[S: Sc](
      in: Int,
      attentionHiddenPerHeadDim: Int,
      attentionNumHeads: Int,
      mlpHiddenDim: Int,
      out: Int,
      dropout: Double,
      tOpt: STenOptions,
      linearized: Boolean
  ): Zipped3 =
    Zipped3(
      // encoder = TransformerEncoderBlock(
      //   in = in,
      //   attentionHiddenPerHeadDim = attentionHiddenPerHeadDim,
      //   attentionNumHeads = attentionNumHeads,
      //   mlpHiddenDim = mlpHiddenDim,
      //   out = out,
      //   dropout = dropout,
      //   tOpt = tOpt,
      //   linearized = linearized,
      //   gptOrder = true,
      //   causalMask = false
      // ),
      decoder1 = TransformerDecoderBlock(
        in = in,
        attentionHiddenPerHeadDim = attentionHiddenPerHeadDim,
        attentionNumHeads = attentionNumHeads,
        mlpHiddenDim = mlpHiddenDim,
        out = out,
        dropout = dropout,
        tOpt = tOpt,
        linearized = linearized,
        decoderDecoderCausalMask = true,
        encoderDecoderCausalMask = true
      ),
      decoder2 = TransformerDecoderBlock(
        in = in,
        attentionHiddenPerHeadDim = attentionHiddenPerHeadDim,
        attentionNumHeads = attentionNumHeads,
        mlpHiddenDim = mlpHiddenDim,
        out = out,
        dropout = dropout,
        tOpt = tOpt,
        linearized = linearized,
        decoderDecoderCausalMask = true,
        encoderDecoderCausalMask = true
      ),
      norm = LayerNorm(List(out.toLong), tOpt)
    )

  implicit val trainingMode: TrainingMode[Zipped3] =
    TrainingMode
      .make[Zipped3](
        m =>
          m.copy(
            // encoder = m.encoder.asEval,
            decoder1 = m.decoder1.asEval,
            decoder2 = m.decoder2.asEval,
            norm = m.norm.asEval
          ),
        m =>
          m.copy(
            // encoder = m.encoder.asTraining,
            decoder1 = m.decoder1.asTraining,
            decoder2 = m.decoder2.asTraining,
            norm = m.norm.asTraining
          )
      )

  implicit val load: Load[Zipped3] =
    Load.compose(
      // _.encoder,
      _.decoder1,
      _.decoder2,
      _.norm
    )
}

case class Zip3Transformer(
    blocks: Seq[Zipped3]
) extends GenericModule[
      (Variable, Variable),
      (Variable, Variable)
    ] {
  def state = blocks.map(_.state).foldLeft(List.empty[(Constant, PTag)])(_ ++ _)
  def forward[S: Sc](
      x: (Variable, Variable)
  ): (Variable, Variable) = {
    val (decoderInput, encoderInput) = x
    blocks.foldLeft((decoderInput, encoderInput)) { (a, block) =>
      block.forward((a._1, a._2))
    }
  }
}

object Zip3Transformer {

  /* Factory of a single transformer block */
  def apply[S: Sc](
      numBlocks: Int,
      in: Int,
      attentionHiddenPerHeadDim: Int,
      attentionNumHeads: Int,
      mlpHiddenDim: Int,
      dropout: Double,
      tOpt: STenOptions,
      linearized: Boolean
  ): Zip3Transformer =
    Zip3Transformer(
      0 until numBlocks map (_ =>
        Zipped3(
          in = in,
          attentionHiddenPerHeadDim = attentionHiddenPerHeadDim,
          attentionNumHeads = attentionNumHeads,
          mlpHiddenDim = mlpHiddenDim,
          out = in,
          dropout = dropout,
          tOpt = tOpt,
          linearized = linearized
        )
      )
    )

  implicit val trainingMode: TrainingMode[Zip3Transformer] =
    TrainingMode
      .make[Zip3Transformer](
        m => m.copy(blocks = m.blocks.map(_.asEval)),
        m => m.copy(blocks = m.blocks.map(_.asTraining))
      )

  implicit val load: Load[Zip3Transformer] = Load.make[Zip3Transformer] {
    m => tensors =>
      m.blocks.foldLeft((List[Unit](), tensors)) {
        case ((acc, params), member) =>
          val numParam = member.state.size
          val loaded = member.load(params.take(numParam))
          (acc.:+(loaded), params.drop(numParam))

      }
  }
}
