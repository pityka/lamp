package lamp.nn.bert

import lamp.autograd.Variable
import lamp.Sc
import lamp.nn._

import lamp.STen
import lamp.autograd.Constant
import lamp.autograd.{param, const}
import lamp.STenOptions

import lamp.autograd.Mean

//
// todo in data: tokenization, masking,

case class BertLossInput(
    input: BertPretrainInput,
    maskedLanguageModelTarget: STen,
    wholeSentenceTarget: STen
)

object BertLossInput {}

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
          output.languageModelScores.flatten(0, 1).logSoftMax(1),
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
      linearized: Boolean
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
      padToken = padToken,
      tOpt = tOpt,
      linearized = linearized
    ),
    mlmLoss = LossFunctions.NLL(
      numClasses = vocabularySize,
      classWeights = STen.ones(List(vocabularySize), tOpt),
      reduction = Mean,
      ignore = padToken
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

case class BertPretrainInput(
    tokens: Variable,
    segments: Variable,
    positions: STen
)

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

    val encoded = encoder.forward((x.tokens, x.segments))
    val mlmScores = mlm.forward((encoded, x.positions))
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
      padToken: Long,
      tOpt: STenOptions,
      linearized: Boolean
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
      padToken = padToken,
      tOpt = tOpt,
      linearized = linearized
    ),
    mlm = MaskedLanguageModelModule(
      inputDim = embeddingDim,
      hiddenDim = mlmHiddenDim,
      vocabularySize = vocabularySize,
      tOpt = tOpt
    ),
    wholeSentenceBinaryClassifier = sequence(
      Linear(embeddingDim, wholeStentenceHiddenDim, tOpt),
      Fun(_ => _.tanh),
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

/** Masked Language Model
  * Input of (embedding, positions)
  * Embedding of size (batch, num tokens, embedding dim)
  * Positions of size (batch, max num tokens) long tensor indicating which positions to make predictions on
  * Output (batch, len(Positions), vocabulary size)
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
      Fun(_ => _.relu),
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
  * Input is `(tokens, segments)` where
  * `tokens` and `segments` are both (batch,num tokens) long tensor.
  *
  * Output is (batch, num tokens, out dimension)
  */
case class BertEncoder(
    tokenEmbedding: Embedding,
    segmentEmbedding: Embedding,
    positionalEmbedding: Constant,
    blocks: Seq[TransformerEncoderBlock]
) extends GenericModule[(Variable, Variable), Variable] {
  def state = tokenEmbedding.state ++ segmentEmbedding.state ++ List(
    positionalEmbedding ->
      BertEncoder.PositionalEmbeddingWeight
  ) ++ blocks.map(_.state).foldLeft(List.empty[(Constant, PTag)])(_ ++ _)
  def forward[S: Sc](x: (Variable, Variable)): Variable = {
    val (tokens, segments) = x
    val embedded = tokenEmbedding.forward(tokens) + segmentEmbedding.forward(
      segments
    ) + positionalEmbedding.slice(
      dim = 1,
      start = 0,
      end = tokens.shape(1),
      step = 1
    )
    blocks.foldLeft(embedded) { (a, block) => block.forward((a, tokens.value)) }
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
    * Input is `(tokens, segments)` where
    * `tokens` and `segments` are both (batch,num tokens) long tensor.
    *
    * @param maxLength maximum num token length
    * @param vocabularySize vocabulary size
    * @param numBlocks number of transformer blocks to create
    * @param embeddingDim input embedding dimension
    * @param attentionHiddenPerHeadDim size of hidden attention dimension of each attention head
    * @param attentionNumHeads number of attention heads
    * @param mlpHiddenDim size of hidden dimension of the two layer perceptron
    * @param out output dimension
    * @param dropout dropout rate
    * @param padToken pad token, (batch, seq) positions where `tokens` == `padToken` are ignored,
    *        padding is not the same as masking
    * @param tOpt tensor options
    * @return a module
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
      padToken: Long,
      tOpt: STenOptions,
      linearized: Boolean
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
      positionalEmbedding =
        param(STen.normal(0d, 1d, List(1, maxLength, embeddingDim), tOpt)),
      blocks = 0 until numBlocks map (_ =>
        TransformerEncoderBlock(
          in = embeddingDim,
          attentionHiddenPerHeadDim = attentionHiddenPerHeadDim,
          attentionNumHeads = attentionNumHeads,
          mlpHiddenDim = mlpHiddenDim,
          out = embeddingDim,
          dropout = dropout,
          padToken = padToken,
          tOpt = tOpt,
          linearized = linearized
        )
      )
    )
}
