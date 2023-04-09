package lamp.nn.bert

import lamp.autograd.Variable
import lamp.Sc
import lamp.nn._

import lamp.STen
import lamp.autograd.Constant
import lamp.autograd.{param, const}
import lamp.STenOptions

import lamp.autograd.Mean
import lamp.SinglePrecision
import lamp.Movable

/** Input to BertLoss module
  *
  *   - input: feature data, see documentation of BertPretrainInput
  *   - maskedLanguageModelTarget: long tensor of (batch size, masked positions
  *     (variable)). Values are the true tokens masked out at the positions in
  *     input.positions
  *   - wholeSentenceTarget: float tensor of size (batch size). Values are truth
  *     targets for the whole sentence loss which is a BCEWithLogitLoss. Values
  *     are floats in [0,1].
  *
  * @param input
  * @param maskedLanguageModelTarget
  * @param wholeSentenceTarget
  */
case class BertLossInput(
    input: BertPretrainInput,
    maskedLanguageModelTarget: STen,
    wholeSentenceTarget: STen
)

object BertLossInput {
  implicit val movable: Movable[BertLossInput] = Movable.by(v =>
    (v.input, v.maskedLanguageModelTarget, v.wholeSentenceTarget)
  )
}

case class BertLoss(
    pretrain: BertPretrainModule,
    mlmLoss: LossFunction,
    wholeSentenceLoss: LossFunction
) extends GenericModule[BertLossInput, Variable] {
  def state = pretrain.state
  def forward[S: Sc](x: BertLossInput): Variable = {
    val output = pretrain.forward(x.input)
    val (l1, _) =
      mlmLoss
        .apply(
          output.languageModelScores.flatten(0, 1),
          x.maskedLanguageModelTarget.view(-1L)
        )

    val (l2, _) = wholeSentenceLoss.apply(
      output.wholeSentenceBinaryClassifierScore,
      x.wholeSentenceTarget
    )
    l1 + l2
  }
}

object BertLoss {

  /** Allocate Bert module
    *
    * @param maxLength
    *   Total sequence length including cls, sep and potential pad tokens
    * @param vocabularySize
    *   Total vocabulary size including cls, sep, pad, mask tokens
    * @param segmentVocabularySize
    *   Vocabulary size of the segment features
    * @param mlmHiddenDim
    *   Hidden dimension of the masked language model decoder
    * @param wholeStentenceHiddenDim
    *   Hidden dimension of the whole sentence task decoder
    * @param numBlocks
    *   Number of transformer blocks
    * @param embeddingDim
    *   Width of the initial embedding dimension, as well as the output width of
    *   the feed forward network in each transformer block
    * @param attentionHiddenPerHeadDim
    * @param attentionNumHeads
    * @param bertEncoderMlpHiddenDim
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
      segmentVocabularySize: Int,
      mlmHiddenDim: Int,
      wholeStentenceHiddenDim: Int,
      numBlocks: Int,
      embeddingDim: Int,
      attentionHiddenPerHeadDim: Int,
      attentionNumHeads: Int,
      bertEncoderMlpHiddenDim: Int,
      dropout: Double,
      padToken: Long,
      tOpt: STenOptions,
      linearized: Boolean,
      positionEmbedding: Option[STen]
  ): BertLoss = BertLoss(
    pretrain = BertPretrainModule(
      maxLength = maxLength,
      vocabularySize = vocabularySize,
      segmentVocabularySize = segmentVocabularySize,
      mlmHiddenDim = mlmHiddenDim,
      wholeStentenceHiddenDim = wholeStentenceHiddenDim,
      numBlocks = numBlocks,
      embeddingDim = embeddingDim,
      attentionNumHeads = attentionNumHeads,
      attentionHiddenPerHeadDim = attentionHiddenPerHeadDim,
      bertEncoderMlpHiddenDim = bertEncoderMlpHiddenDim,
      dropout = dropout,
      tOpt = tOpt,
      linearized = linearized,
      positionEmbedding = positionEmbedding
    ),
    mlmLoss = LossFunctions.NLL(
      numClasses = vocabularySize,
      classWeights = STen.ones(List(vocabularySize), tOpt),
      reduction = Mean,
      ignore = padToken // not sure this is needed 
    ),
    wholeSentenceLoss = LossFunctions.BCEWithLogits(
      posWeights = None,
      reduction = Mean,
      ignore = -100L
    )
  )

  implicit val trainingMode: TrainingMode[BertLoss] = TrainingMode
    .make[BertLoss](
      m =>
        m.copy(
          pretrain = m.pretrain.asEval
        ),
      m =>
        m.copy(
          pretrain = m.pretrain.asTraining
        )
    )
  implicit val load: Load[BertLoss] =
    Load.compose(_.pretrain)

}

/** Input for BERT pretrain module
  *
  *   - Tokens: Long tensor of size (batch, sequence length). Sequence length
  *     includes cls and sep tokens. Values are tokens of the input vocabulary
  *     and 4 additional control tokens: cls, sep, pad, mask. First token must
  *     be cls.
  *   - Segments: Long tensor of size (batch, sequence length). Values are
  *     segment tokens.
  *   - Positions: Long tensor of size (batch, mask size (variable)). Values are
  *     indices in [0,sequence length) selecting masked sequence positions. They
  *     never select positions of cls, sep, pad.
  *   - maxLength: 1D long tensor of size (sequence length). Values are in [0,sequence_length]. 
  *     Tokens at positions higher or equal than the sequence length are ignored.
  *
  * @param tokens
  * @param segments
  * @param positions
  */
case class BertPretrainInput(
    tokens: Constant,
    segments: Constant,
    positions: STen,
    maxLength: Option[STen]
)

object BertPretrainInput {
  implicit val movable: Movable[BertPretrainInput] =
    Movable.by(v => (v.tokens, v.segments, v.positions, v.maxLength))
}

/** Output of BERT
  *
  *   - encoded: float tensor of size (batch, sequence length, embedding
  *     dimension ) holds per token embeddings
  *   - languageModelScores: float tensor of size (batch, sequence length,
  *     vocabulary size) holds per token log probability distributions (from
  *     logSoftMax)
  *   - wholeSentenceBinaryClassifierScore: float tensor of size (batch) holds
  *     the output score of the whole sentence prediction task suitable for
  *     BCELogitLoss
  *
  * @param encoded
  * @param languageModelScores
  * @param wholeSentenceBinaryClassifierScore
  */
case class BertPretrainOutput(
    encoded: Variable,
    languageModelScores: Variable,
    wholeSentenceBinaryClassifierScore: Variable
)

case class BertPretrainModule(
    encoder: BertEncoder,
    mlm: MaskedLanguageModelModule,
    wholeSentenceBinaryClassifier: BertPretrainModule.MLP
) extends GenericModule[BertPretrainInput, BertPretrainOutput] {
  def state = encoder.state ++ mlm.state ++ wholeSentenceBinaryClassifier.state

  def forward[S: Sc](x: BertPretrainInput): BertPretrainOutput = {

    val encoded = encoder.forward((x.tokens, x.segments, x.maxLength))
    val mlmScores = mlm.forward((encoded, x.positions)).logSoftMax(dim = 2)
    val encodedClsPosition = encoded.select(dim = 1, index = 0)
    val binaryScore =
      wholeSentenceBinaryClassifier.forward(encodedClsPosition).view(List(-1L))
    BertPretrainOutput(
      encoded = encoded,
      languageModelScores = mlmScores,
      wholeSentenceBinaryClassifierScore = binaryScore
    )
  }

}

object BertPretrainModule {

  type MLP = Seq3[
    Variable,
    Variable,
    Variable,
    Variable,
    Linear,
    Fun,
    Linear
  ]

  def apply[S: Sc](
      maxLength: Int,
      vocabularySize: Int,
      segmentVocabularySize: Int,
      mlmHiddenDim: Int,
      wholeStentenceHiddenDim: Int,
      numBlocks: Int,
      embeddingDim: Int,
      attentionHiddenPerHeadDim: Int,
      attentionNumHeads: Int,
      bertEncoderMlpHiddenDim: Int,
      dropout: Double,
      tOpt: STenOptions,
      linearized: Boolean,
      positionEmbedding: Option[STen]
  ): BertPretrainModule = BertPretrainModule(
    encoder = BertEncoder(
      maxLength = maxLength,
      vocabularySize = vocabularySize,
      segmentVocabularySize = segmentVocabularySize,
      numBlocks = numBlocks,
      embeddingDim = embeddingDim,
      attentionHiddenPerHeadDim = attentionHiddenPerHeadDim,
      attentionNumHeads = attentionNumHeads,
      mlpHiddenDim = bertEncoderMlpHiddenDim,
      dropout = dropout,
      tOpt = tOpt,
      linearized = linearized,
      positionEmbedding = positionEmbedding
    ),
    mlm = MaskedLanguageModelModule(
      inputDim = embeddingDim,
      hiddenDim = mlmHiddenDim,
      vocabularySize = vocabularySize,
      tOpt = tOpt
    ),
    wholeSentenceBinaryClassifier = sequence(
      Linear(embeddingDim, wholeStentenceHiddenDim, tOpt),
      Fun(scope => _.tanh(scope)),
      Linear(wholeStentenceHiddenDim, 1, tOpt)
    )
  )

  implicit val trainingMode: TrainingMode[BertPretrainModule] = TrainingMode
    .make[BertPretrainModule](
      m =>
        m.copy(
          encoder = m.encoder.asEval,
          mlm = m.mlm.asEval
        ),
      m =>
        m.copy(
          encoder = m.encoder.asTraining,
          mlm = m.mlm.asTraining
        )
    )
  implicit val load: Load[BertPretrainModule] =
    Load.compose(_.encoder, _.mlm, _.wholeSentenceBinaryClassifier)

}

/** Masked Language Model Input of (embedding, positions) Embedding of size
  * (batch, num tokens, embedding dim) Positions of size (batch, max num tokens)
  * long tensor indicating which positions to make predictions on Output (batch,
  * len(Positions), vocabulary size)
  *
  * @param mlp
  */
case class MaskedLanguageModelModule(
    mlp: MaskedLanguageModelModule.MLP
) extends GenericModule[(Variable, STen), Variable] {
  def state = mlp.state

  def forward[S: Sc](x: (Variable, STen)): Variable = {
    val (encoderOutput, predictionPositions) = x
    val encoderOutputAtPredictionPositions =
      encoderOutput
        .view(List(-1, encoderOutput.shape(2)))
        .indexSelect(dim = 0, index = const(predictionPositions.view(-1)))
        .view(
          List(
            encoderOutput.shape(0),
            predictionPositions.shape(1),
            encoderOutput.shape(2)
          )
        )
    mlp.forward(encoderOutputAtPredictionPositions)
  }
}
object MaskedLanguageModelModule {

  type MLP = Seq4[
    Variable,
    Variable,
    Variable,
    Variable,
    Variable,
    Linear,
    Fun,
    LayerNorm,
    Linear
  ]

  def apply[S: Sc](
      inputDim: Int,
      hiddenDim: Int,
      vocabularySize: Int,
      tOpt: STenOptions
  ): MaskedLanguageModelModule = MaskedLanguageModelModule(
    sequence(
      Linear(inputDim, hiddenDim, tOpt),
      Fun(scope => _.relu(scope)),
      LayerNorm(List(hiddenDim), tOpt),
      Linear(hiddenDim, vocabularySize, tOpt)
    )
  )
  implicit val trainingMode: TrainingMode[MaskedLanguageModelModule] =
    TrainingMode
      .make[MaskedLanguageModelModule](
        m =>
          m.copy(
            m.mlp.asEval
          ),
        m =>
          m.copy(
            m.mlp.asTraining
          )
      )
  implicit val load: Load[MaskedLanguageModelModule] =
    Load.make[MaskedLanguageModelModule] { m => tensors =>
      m.mlp.load(tensors)
    }
}

/** BertEncoder module
  *
  * Input is `(tokens, segments, maxLength)` where `tokens` and `segments` are both
  * (batch,num tokens) long tensor. maxLength is a 1D long tensor indicating the length of input sequences
  *
  * Output is (batch, num tokens, out dimension)
  */
case class BertEncoder(
    tokenEmbedding: Embedding,
    segmentEmbedding: Embedding,
    positionalEmbedding: Constant,
    blocks: Seq[TransformerEncoderBlock]
) extends GenericModule[(Variable, Variable, Option[STen]), Variable] {
  def state = tokenEmbedding.state ++ segmentEmbedding.state ++ List(
    positionalEmbedding ->
      BertEncoder.PositionalEmbeddingWeight
  ) ++ blocks.map(_.state).foldLeft(List.empty[(Constant, PTag)])(_ ++ _)
  def forward[S: Sc](x: (Variable, Variable, Option[STen])): Variable = {
    val (tokens, segments, maxLength) = x
    val embedded = tokenEmbedding.forward(tokens) + segmentEmbedding.forward(
      segments
    ) + positionalEmbedding.slice(
      dim = 1,
      start = 0,
      end = tokens.shape(1),
      step = 1
    )
    blocks.foldLeft(embedded) { (a, block) => block.forward((a, maxLength)) }
  }
}
object BertEncoder {
  object PositionalEmbeddingWeight extends LeafTag
  implicit val trainingMode: TrainingMode[BertEncoder] = TrainingMode
    .make[BertEncoder](
      m =>
        m.copy(
          blocks = m.blocks.map(_.asEval),
          tokenEmbedding = m.tokenEmbedding.asEval,
          segmentEmbedding = m.segmentEmbedding.asEval
        ),
      m =>
        m.copy(
          blocks = m.blocks.map(_.asTraining),
          tokenEmbedding = m.tokenEmbedding.asTraining,
          segmentEmbedding = m.segmentEmbedding.asTraining
        )
    )
  implicit val load: Load[BertEncoder] = Load.make[BertEncoder] {
    m => tensors =>
      m.tokenEmbedding.load(tensors.take(m.tokenEmbedding.state.size))
      m.segmentEmbedding.load(
        tensors
          .drop(m.tokenEmbedding.state.size)
          .take(m.segmentEmbedding.state.size)
      )
      m.positionalEmbedding.value.copyFrom(
        tensors
          .drop(m.tokenEmbedding.state.size + m.segmentEmbedding.state.size)
          .head
      )
      m.blocks.foldLeft(
        (
          List[Unit](),
          tensors.drop(
            m.tokenEmbedding.state.size + m.segmentEmbedding.state.size + 1
          )
        )
      ) { case ((acc, params), member) =>
        val numParam = member.state.size
        val loaded = member.load(params.take(numParam))
        (acc.:+(loaded), params.drop(numParam))

      }

  }

  /** Factory for the encoder module of Bert
    *
    * Input is `(tokens, segments)` where `tokens` and `segments` are both
    * (batch,num tokens) long tensor.
    *
    * @param maxLength
    *   maximum num token length
    * @param vocabularySize
    *   vocabulary size
    * @param numBlocks
    *   number of transformer blocks to create
    * @param embeddingDim
    *   input embedding dimension
    * @param attentionHiddenPerHeadDim
    *   size of hidden attention dimension of each attention head
    * @param attentionNumHeads
    *   number of attention heads
    * @param mlpHiddenDim
    *   size of hidden dimension of the two layer perceptron
    * @param out
    *   output dimension
    * @param dropout
    *   dropout rate
    * @param tOpt
    *   tensor options
    * @param positionEmbedding optional float tensor of size (sequence length, embedding dimension)
    *   if missing the absolute positional embeddings from Vaswani et al 2017 is used
    *   Following the Bert paper the position embeddings are summed
    * @return
    *   a module
    */
  def apply[S: Sc](
      maxLength: Int,
      vocabularySize: Int,
      segmentVocabularySize: Int,
      numBlocks: Int,
      embeddingDim: Int,
      attentionHiddenPerHeadDim: Int,
      attentionNumHeads: Int,
      mlpHiddenDim: Int,
      dropout: Double,
      tOpt: STenOptions,
      linearized: Boolean,
      positionEmbedding: Option[STen]
  ): BertEncoder =
    BertEncoder(
      tokenEmbedding = Embedding.apply(
        classes = vocabularySize,
        dimensions = embeddingDim,
        tOpt = tOpt
      ),
      segmentEmbedding = Embedding.apply(
        classes = segmentVocabularySize,
        dimensions = embeddingDim,
        tOpt = tOpt
      ),
      positionalEmbedding = param(
        positionEmbedding.getOrElse(PositionalEmbedding
          .vaswani(
            sequenceLength = maxLength,
            dimension = embeddingDim,
            device = lamp.Device.fromOptions(tOpt),
            SinglePrecision
          ))
          .unsqueeze(0)
      ),
      blocks = 0 until numBlocks map (_ =>
        TransformerEncoderBlock(
          in = embeddingDim,
          attentionHiddenPerHeadDim = attentionHiddenPerHeadDim,
          attentionNumHeads = attentionNumHeads,
          mlpHiddenDim = mlpHiddenDim,
          out = embeddingDim,
          dropout = dropout,
          tOpt = tOpt,
          linearized = linearized
        )
      )
    )
}
