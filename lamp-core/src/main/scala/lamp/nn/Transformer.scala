package lamp.nn

import lamp.autograd.Variable
import lamp.Sc

import lamp.STen
import lamp.autograd.Constant
import lamp.autograd.{param, const}
import lamp.STenOptions
import lamp.Scope
import lamp.FloatingPointPrecision
import lamp.Device
import lamp.autograd.ScaledDotProductAttention

/** TransformerEncoder module
  *
  * Does *not* include initial embedding or position encoding.
  *
  * Input is `(data, maxLength)` where `data` is (batch, sequence, input
  * dimension), double tensor `maxLength` is a 1D or 2D long tensor used for
  * attention masking.
  *
  * Attention masking is implemented similarly to chapter 11.3.2.1 in d2l.ai
  * v1.0.0-beta0. It supports unmasked attention, attention on variable length
  * input, and left-to-right attention.
  *
  * Output is (bach, sequence, output dimension)
  */
case class TransformerEncoder(
    blocks: Seq[TransformerEncoderBlock]
) extends GenericModule[(Variable, Option[STen]), Variable] {
  def state = blocks.map(_.state).foldLeft(List.empty[(Constant, PTag)])(_ ++ _)
  def forward[S: Sc](x: (Variable, Option[STen])): Variable = {
    val (input, maxLength) = x
    blocks.foldLeft(input) { (a, block) => block.forward((a, maxLength)) }
  }
}
object TransformerEncoder {
  implicit val trainingMode: TrainingMode[TransformerEncoder] = TrainingMode
    .make[TransformerEncoder](
      m => m.copy(blocks = m.blocks.map(_.asEval)),
      m => m.copy(blocks = m.blocks.map(_.asTraining))
    )
  implicit val load: Load[TransformerEncoder] = Load.make[TransformerEncoder] {
    m => tensors =>
      m.blocks.foldLeft((List[Unit](), tensors)) {
        case ((acc, params), member) =>
          val numParam = member.state.size
          val loaded = member.load(params.take(numParam))
          (acc.:+(loaded), params.drop(numParam))

      }

  }

  /** Factory for the encoder module of transformer. Does *not* include initial
    * embedding or position encoding.
    *
    * @param numBlocks
    *   number of transformer blocks to create (layers)
    * @param in
    *   input dimension
    * @param attentionHiddenPerHeadDim
    *   size of hidden attention dimension of each attention head
    * @param attentionNumHeads
    *   number of attention heads
    * @param mlpHiddenDim
    *   size of hidden dimension of the two layer perceptron
    * @param dropout
    *   dropout rate
    * @param tOpt
    *   tensor options
    * @return
    *   a module
    */
  def apply[S: Sc](
      numBlocks: Int,
      in: Int,
      attentionHiddenPerHeadDim: Int,
      attentionNumHeads: Int,
      mlpHiddenDim: Int,
      dropout: Double,
      tOpt: STenOptions,
      linearized: Boolean,
      gptOrder: Boolean,
      causalMask: Boolean
  ): TransformerEncoder =
    TransformerEncoder(
      0 until numBlocks map (_ =>
        TransformerEncoderBlock(
          in = in,
          attentionHiddenPerHeadDim = attentionHiddenPerHeadDim,
          attentionNumHeads = attentionNumHeads,
          mlpHiddenDim = mlpHiddenDim,
          out = in,
          dropout = dropout,
          tOpt = tOpt,
          linearized = linearized,
          gptOrder = gptOrder,
          causalMask = causalMask
        )
      )
    )
}
case class TransformerDecoder(
    blocks: Seq[TransformerDecoderBlock]
) extends GenericModule[(Variable, Variable, Option[STen]), Variable] {
  def state = blocks.map(_.state).foldLeft(List.empty[(Constant, PTag)])(_ ++ _)
  def forward[S: Sc](x: (Variable, Variable, Option[STen])): Variable = {
    val (input, encoderOutput, maxLength) = x
    blocks.foldLeft(input) { (a, block) =>
      block.forward((a, encoderOutput, maxLength))
    }
  }
}
object TransformerDecoder {
  implicit val trainingMode: TrainingMode[TransformerDecoder] = TrainingMode
    .make[TransformerDecoder](
      m => m.copy(blocks = m.blocks.map(_.asEval)),
      m => m.copy(blocks = m.blocks.map(_.asTraining))
    )
  implicit val load: Load[TransformerDecoder] = Load.make[TransformerDecoder] {
    m => tensors =>
      m.blocks.foldLeft((List[Unit](), tensors)) {
        case ((acc, params), member) =>
          val numParam = member.state.size
          val loaded = member.load(params.take(numParam))
          (acc.:+(loaded), params.drop(numParam))

      }

  }

  /** Factory for the decoder module of transformer. Does *not* include initial
    * embedding or position encoding.
    *
    * @param numBlocks
    *   number of transformer blocks to create (layers)
    * @param in
    *   input dimension
    * @param attentionHiddenPerHeadDim
    *   size of hidden attention dimension of each attention head
    * @param attentionNumHeads
    *   number of attention heads
    * @param mlpHiddenDim
    *   size of hidden dimension of the two layer perceptron
    * @param dropout
    *   dropout rate
    * @param tOpt
    *   tensor options
    * @return
    *   a module
    */
  def apply[S: Sc](
      numBlocks: Int,
      in: Int,
      attentionHiddenPerHeadDim: Int,
      attentionNumHeads: Int,
      mlpHiddenDim: Int,
      dropout: Double,
      tOpt: STenOptions,
      linearized: Boolean,
      decoderDecoderCausalMask: Boolean,
      encoderDecoderCausalMask: Boolean
  ): TransformerDecoder =
    TransformerDecoder(
      0 until numBlocks map (_ =>
        TransformerDecoderBlock(
          in = in,
          attentionHiddenPerHeadDim = attentionHiddenPerHeadDim,
          attentionNumHeads = attentionNumHeads,
          mlpHiddenDim = mlpHiddenDim,
          out = in,
          dropout = dropout,
          tOpt = tOpt,
          linearized = linearized,
          decoderDecoderCausalMask = decoderDecoderCausalMask,
          encoderDecoderCausalMask = encoderDecoderCausalMask
        )
      )
    )
}

/** A single block of the transformer self attention encoder using GELU
  *
  * Input is `(data, maxLength)` where `data` is (batch, sequence, input
  * dimension), double tensor `maxLength` is a 1D or 2D long tensor used for
  * attention masking.
  *
  * The order of operations depends on gptOrder param. If `gptOrder` is true
  * then:
  *   - y = attention(norm(input))+input
  *   - result = mlp(norm(y))+y
  *   - Note that in this case there is no normalization at the end of the
  *     transformer. One may wants to add one separately. This is how GPT2 is
  *     defined in hugging face or nanoGPT.
  *   - Note that the residual connection has a path which does not flow through
  *     the normalization.
  *   - + dimension wise learnable scale parameter in each residual path
  *
  * If `gptOrder` is false then:
  *   - y = norm(attention(input)+input )
  *   - result = norm(mlp(y)+y)
  *   - This follows chapter 11.7 in d2l.ai v1.0.0-beta0. (Same as in
  *     https://arxiv.org/pdf/1706.03762.pdf)
  *   - Note that the residual connection has a path which flows through the
  *     normalization.
  *
  * Output is (bach, sequence, output dimension)
  */
case class TransformerEncoderBlock(
    attention: MultiheadAttention,
    layerNorm1: LayerNorm,
    layerNorm2: LayerNorm,
    w1: Constant,
    b1: Constant,
    w2: Constant,
    b2: Constant,
    scale1: Constant,
    scale2: Constant,
    dropout: Double,
    train: Boolean,
    gptOrder: Boolean
) extends GenericModule[(Variable, Option[STen]), Variable] {

  def state =
    attention.state ++ layerNorm1.state ++ layerNorm2.state ++ Seq(
      w1 -> TransformerEncoderBlock.Weights1,
      w2 -> TransformerEncoderBlock.Weights2,
      b1 -> TransformerEncoderBlock.Bias1,
      b2 -> TransformerEncoderBlock.Bias2,
      scale1 -> TransformerEncoderBlock.Scale1,
      scale2 -> TransformerEncoderBlock.Scale2
    )

  def forward[S: Sc](x: (Variable, Option[STen])): Variable = {

    def mm1(a: Variable, b: Variable) = {
      val shape = a.shape
      a.view(List(-1, shape.last)).mm(b).view(shape.dropRight(1) :+ -1L)
    }

    val (input, maxLength) = x
    if (gptOrder) {
      val a1 = layerNorm1(input.dropout(dropout, train))
      val a2 = attention.forward((a1, a1, a1, maxLength)) * scale1 + input
      val a3 = layerNorm2(a2.dropout(dropout, train))
      val a4 = (mm1((mm1(a3, w1) + b1).gelu, w2) + b2) * scale2 + a2

      a4
    } else {
      val a1 = attention.forward((input, input, input, maxLength))
      val a2 = layerNorm1(a1.dropout(dropout, train) + input)
      val a3 = mm1((mm1(a2, w1) + b1).gelu, w2) + b2

      val a4 = layerNorm2(a3.dropout(dropout, train) + a3)
      a4
    }
  }

}
case class TransformerDecoderBlock(
    attentionDecoderDecoder: MultiheadAttention,
    attentionEncoderDecoder: MultiheadAttention,
    layerNorm1: LayerNorm,
    layerNorm2: LayerNorm,
    layerNorm3: LayerNorm,
    layerNorm4: LayerNorm,
    w1: Constant,
    b1: Constant,
    w2: Constant,
    b2: Constant,
    dropout: Double,
    train: Boolean
) extends GenericModule[(Variable, Variable, Option[STen]), Variable] {

  def state =
    attentionDecoderDecoder.state ++ attentionEncoderDecoder.state ++ layerNorm1.state ++ layerNorm2.state ++ layerNorm3.state ++ layerNorm4.state ++ Seq(
      w1 -> TransformerEncoderBlock.Weights1,
      w2 -> TransformerEncoderBlock.Weights2,
      b1 -> TransformerEncoderBlock.Bias1,
      b2 -> TransformerEncoderBlock.Bias2
    )

  def forward[S: Sc](x: (Variable, Variable, Option[STen])): Variable = {

    def mm1(a: Variable, b: Variable) = {
      val shape = a.shape
      a.view(List(-1, shape.last)).mm(b).view(shape.dropRight(1) :+ -1L)
    }

    val (decoderInput, encoderOutput, maxLength) = x
    val a1 = layerNorm1(decoderInput.dropout(dropout, train))
    val a2 =
      attentionDecoderDecoder.forward((a1, a1, a1, maxLength)) + decoderInput
    val a3 = layerNorm2(a2.dropout(dropout, train))
    val a4 = layerNorm3(encoderOutput.dropout(dropout, train))
    val a5 = a2 + attentionEncoderDecoder.forward((a3, a4, a4, None))

    val a6 = layerNorm4(a5.dropout(dropout, train))
    val a7 = mm1((mm1(a6, w1) + b1).gelu, w2) + b2 + a5

    a7

  }

}

case class Transformer(
    encoder: TransformerEncoder,
    decoder: TransformerDecoder
) extends GenericModule[
      (Variable, Variable, Option[STen], Option[STen]),
      Variable
    ] {
  def state = encoder.state ++ decoder.state
  def forward[S: Sc](
      x: (Variable, Variable, Option[STen], Option[STen])
  ): Variable = {
    val (decoderInput, encoderInput, decoderMaxLength, encoderMaxLength) = x
    val encoderOutput = encoder.forward((encoderInput, encoderMaxLength))
    decoder.forward((decoderInput, encoderOutput, decoderMaxLength))
  }
}

object Transformer {

  /* Factory of a single transformer block */
  def apply[S: Sc](
      numBlocks: Int,
      in: Int,
      attentionHiddenPerHeadDim: Int,
      attentionNumHeads: Int,
      mlpHiddenDim: Int,
      dropout: Double,
      tOpt: STenOptions,
      linearized: Boolean,
      encoderCausalMask: Boolean = false,
      decoderDecoderCausalMask: Boolean = true,
      encoderDecoderCausalMask: Boolean = false
  ): Transformer =
    Transformer(
      TransformerEncoder(
        numBlocks = numBlocks,
        in = in,
        attentionHiddenPerHeadDim = attentionHiddenPerHeadDim,
        attentionNumHeads = attentionNumHeads,
        mlpHiddenDim = mlpHiddenDim,
        dropout = dropout,
        tOpt = tOpt,
        linearized = linearized,
        gptOrder = true,
        causalMask = encoderCausalMask
      ),
      TransformerDecoder(
        numBlocks = numBlocks,
        in = in,
        attentionHiddenPerHeadDim = attentionHiddenPerHeadDim,
        attentionNumHeads = attentionNumHeads,
        mlpHiddenDim = mlpHiddenDim,
        dropout = dropout,
        tOpt = tOpt,
        linearized = linearized,
        decoderDecoderCausalMask = decoderDecoderCausalMask,
        encoderDecoderCausalMask = encoderDecoderCausalMask
      )
    )

  implicit val trainingMode: TrainingMode[Transformer] =
    TrainingMode
      .make[Transformer](
        m => m.copy(encoder = m.encoder.asEval, decoder = m.decoder.asEval),
        m =>
          m.copy(encoder = m.encoder.asTraining, decoder = m.decoder.asTraining)
      )

  implicit val load: Load[Transformer] = Load.compose(_.encoder, _.decoder)
}

object TransformerDecoderBlock {

  /* Factory of a single transformer block */
  def apply[S: Sc](
      in: Int,
      attentionHiddenPerHeadDim: Int,
      attentionNumHeads: Int,
      mlpHiddenDim: Int,
      out: Int,
      dropout: Double,
      tOpt: STenOptions,
      linearized: Boolean,
      decoderDecoderCausalMask: Boolean,
      encoderDecoderCausalMask: Boolean
  ): TransformerDecoderBlock =
    TransformerDecoderBlock(
      attentionDecoderDecoder = MultiheadAttention(
        dQ = in,
        dK = in,
        dV = in,
        hiddenPerHead = attentionHiddenPerHeadDim,
        out = in,
        numHeads = attentionNumHeads,
        dropout = dropout,
        tOpt = tOpt,
        linearized = linearized,
        causalMask = decoderDecoderCausalMask
      ),
      attentionEncoderDecoder = MultiheadAttention(
        dQ = in,
        dK = in,
        dV = in,
        hiddenPerHead = attentionHiddenPerHeadDim,
        out = in,
        numHeads = attentionNumHeads,
        dropout = dropout,
        tOpt = tOpt,
        linearized = linearized,
        causalMask = encoderDecoderCausalMask
      ),
      layerNorm1 = lamp.nn.LayerNorm(List(in.toLong), tOpt),
      layerNorm2 = lamp.nn.LayerNorm(List(in.toLong), tOpt),
      layerNorm3 = lamp.nn.LayerNorm(List(in.toLong), tOpt),
      layerNorm4 = lamp.nn.LayerNorm(List(in.toLong), tOpt),
      w1 = initLinear(in, mlpHiddenDim, tOpt),
      b1 = param(STen.zeros(List(1, mlpHiddenDim), tOpt)),
      w2 = initLinear(mlpHiddenDim, out, tOpt),
      b2 = param(STen.zeros(List(1, out), tOpt)),
      dropout = dropout,
      train = true
    )

  object Weights1 extends LeafTag
  object Weights2 extends LeafTag
  object Bias1 extends LeafTag
  object Bias2 extends LeafTag

  implicit val trainingMode: TrainingMode[TransformerDecoderBlock] =
    TrainingMode
      .make[TransformerDecoderBlock](
        m =>
          m.copy(
            train = false,
            attentionEncoderDecoder = m.attentionEncoderDecoder.asEval,
            attentionDecoderDecoder = m.attentionDecoderDecoder.asEval
          ),
        m =>
          m.copy(
            train = true,
            attentionEncoderDecoder = m.attentionEncoderDecoder.asTraining,
            attentionDecoderDecoder = m.attentionDecoderDecoder.asTraining
          )
      )

  implicit val load: Load[TransformerDecoderBlock] =
    Load.make[TransformerDecoderBlock] { m => tensors =>
      m.attentionDecoderDecoder.load(tensors)
      m.attentionEncoderDecoder.load(
        tensors.drop(m.attentionDecoderDecoder.state.size)
      )
      val attenionStatesSize =
        m.attentionDecoderDecoder.state.size + m.attentionEncoderDecoder.state.size
      m.layerNorm1.load(tensors.drop(attenionStatesSize))
      m.layerNorm2.load(
        tensors.drop(attenionStatesSize + m.layerNorm1.state.size)
      )
      m.layerNorm3.load(
        tensors.drop(
          attenionStatesSize + m.layerNorm1.state.size + m.layerNorm2.state.size
        )
      )
      m.layerNorm4.load(
        tensors.drop(
          attenionStatesSize + m.layerNorm1.state.size + m.layerNorm2.state.size + m.layerNorm3.state.size
        )
      )

      val remaining = tensors.drop(
        attenionStatesSize + m.layerNorm1.state.size + m.layerNorm2.state.size + m.layerNorm3.state.size + m.layerNorm4.state.size
      )
      m.w1.value.copyFrom(remaining(0))
      m.w2.value.copyFrom(remaining(1))
      m.b1.value.copyFrom(remaining(2))
      m.b2.value.copyFrom(remaining(3))

    }
}

object TransformerEncoderBlock {

  /* Factory of a single transformer block */
  def apply[S: Sc](
      in: Int,
      attentionHiddenPerHeadDim: Int,
      attentionNumHeads: Int,
      mlpHiddenDim: Int,
      out: Int,
      dropout: Double,
      tOpt: STenOptions,
      linearized: Boolean,
      gptOrder: Boolean,
      causalMask: Boolean
  ): TransformerEncoderBlock =
    TransformerEncoderBlock(
      attention = MultiheadAttention(
        dQ = in,
        dK = in,
        dV = in,
        hiddenPerHead = attentionHiddenPerHeadDim,
        out = in,
        numHeads = attentionNumHeads,
        dropout = dropout,
        tOpt = tOpt,
        linearized = linearized,
        causalMask = causalMask
      ),
      gptOrder = gptOrder,
      layerNorm1 = lamp.nn.LayerNorm(List(in.toLong), tOpt),
      layerNorm2 = lamp.nn.LayerNorm(List(in.toLong), tOpt),
      w1 = initLinear(in, mlpHiddenDim, tOpt),
      b1 = param(STen.zeros(List(1, mlpHiddenDim), tOpt)),
      w2 = initLinear(mlpHiddenDim, out, tOpt),
      b2 = param(STen.zeros(List(1, out), tOpt)),
      scale1 = param(STen.normal(0d, 0.0001, List(in.toLong), tOpt)),
      scale2 = param(STen.normal(0d, 0.0001, List(in.toLong), tOpt)),
      dropout = dropout,
      train = true
    )

  object Weights1 extends LeafTag
  object Weights2 extends LeafTag
  object Bias1 extends LeafTag
  object Bias2 extends LeafTag
  object Scale1 extends LeafTag
  object Scale2 extends LeafTag

  implicit val trainingMode: TrainingMode[TransformerEncoderBlock] =
    TrainingMode
      .make[TransformerEncoderBlock](
        m => m.copy(train = false, attention = m.attention.asEval),
        m => m.copy(train = true, attention = m.attention.asTraining)
      )
  implicit val load: Load[TransformerEncoderBlock] =
    Load.make[TransformerEncoderBlock] { m => tensors =>
      m.attention.load(tensors)
      m.layerNorm1.load(tensors.drop(m.attention.state.size))
      m.layerNorm2.load(
        tensors.drop(m.attention.state.size + m.layerNorm1.state.size)
      )

      val remaining = tensors.drop(
        m.attention.state.size + m.layerNorm1.state.size + m.layerNorm2.state.size
      )
      m.w1.value.copyFrom(remaining(0))
      m.w2.value.copyFrom(remaining(1))
      m.b1.value.copyFrom(remaining(2))
      m.b2.value.copyFrom(remaining(3))
      m.scale1.value.copyFrom(remaining(4))
      m.scale2.value.copyFrom(remaining(5))

    }
}

/** Multi-head scaled dot product attention module
  *
  * Input: (query,key,value,maxLength) where
  *   - query: batch x num queries x query dim
  *   - key: batch x num k-v x key dim
  *   - value: batch x num k-v x key value
  *   - maxLength: 1D or 2D long tensor for attention masking
  */
case class MultiheadAttention(
    wQ: Constant,
    wK: Constant,
    wV: Constant,
    wO: Constant,
    dropout: Double,
    train: Boolean,
    numHeads: Int,
    linearized: Boolean,
    causalMask: Boolean
) extends GenericModule[
      (Variable, Variable, Variable, Option[STen]),
      Variable
    ] {

  override val state = List(
    wQ -> MultiheadAttention.WeightsQ,
    wK -> MultiheadAttention.WeightsK,
    wV -> MultiheadAttention.WeightsV,
    wO -> MultiheadAttention.WeightsO
  )

  override def forward[S: Sc](
      x: (Variable, Variable, Variable, Option[STen])
  ): Variable = {
    val (q, k, v, maxLength) = x

    MultiheadAttention.multiheadAttention(
      query = q,
      keys = k,
      values = v,
      maxLength = maxLength,
      dropout = dropout,
      trainDropout = train,
      wQuery = wQ,
      wKeys = wK,
      wValues = wV,
      wOutput = wO,
      numHeads = numHeads,
      linearized = linearized,
      causalMask = causalMask
    )
  }

}

object MultiheadAttention {
  def apply[S: Sc](
      dQ: Int,
      dK: Int,
      dV: Int,
      hiddenPerHead: Int,
      out: Int,
      dropout: Double,
      numHeads: Int,
      tOpt: STenOptions,
      linearized: Boolean,
      causalMask: Boolean
  ): MultiheadAttention = MultiheadAttention(
    wQ = initLinear(dQ, hiddenPerHead * numHeads, tOpt),
    wK = initLinear(dK, hiddenPerHead * numHeads, tOpt),
    wV = initLinear(dV, hiddenPerHead * numHeads, tOpt),
    wO = initLinear(hiddenPerHead * numHeads, out, tOpt),
    dropout = dropout,
    train = true,
    numHeads = numHeads,
    linearized = linearized,
    causalMask = causalMask
  )
  case object WeightsQ extends LeafTag
  case object WeightsK extends LeafTag
  case object WeightsV extends LeafTag
  case object WeightsO extends LeafTag

  implicit val trainingMode: TrainingMode[MultiheadAttention] =
    TrainingMode.identity[MultiheadAttention]
  implicit val load: Load[MultiheadAttention] =
    Load.make[MultiheadAttention](m =>
      tensors => {
        m.wQ.value.copyFrom(tensors.head)
        m.wK.value.copyFrom(tensors(1))
        m.wV.value.copyFrom(tensors(2))
        m.wO.value.copyFrom(tensors(3))

      }
    )

  /** Masks on the 3rd axis of maskable depending on the dimensions of maxLength
    *
    * if maxLength is 2D: (batch,query,key) locations where
    * maxLength(batch,query) > key are ignored.
    *
    * if maxLength is 1D: (batch,query,key) locations where maxLength(batch) >
    * query are ignored
    */
  def sequenceMask[S: Sc](
      maxLength: STen,
      maskable: Variable,
      fill: Double
  ) = {
    if (maxLength.shape.size == 2)
      sequenceMaskValidLength2D(maxLength, maskable, fill)
    else sequenceMaskValidLength1D(maxLength, maskable, fill)

  }

  /** Masks the maskable(i,j,k) cell iff k >= maxLength(i,j)
    *
    * Masks some elements on the last (3rd) axis of maskable
    *
    * @param maxLength
    *   batch x seq, type Long
    * @param maskable
    *   batch x seq x ???
    * @param fill
    *   scalar
    */
  def sequenceMaskValidLength2D[S: Sc](
      maxLength: STen,
      maskable: Variable,
      fill: Double
  ) = {
    assert(maxLength.shape(1) == maskable.shape(1))
    assert(maxLength.shape(0) == maskable.shape(0))
    assert(maxLength.shape.size == 2)
    val mask = STen
      .arange_l(
        start = 0L,
        end = maskable.shape(2),
        step = 1L,
        tensorOptions = maskable.options
      )
      .view(1, 1, -1) ge maxLength.unsqueeze(2)

    maskable.maskFill(
      lamp.autograd.const(mask),
      fill
    )

  }

  /** Masks the maskable(i,j,k) cell iff k >= maxLength(i)
    *
    * @param maxLength
    *   batch, type Long
    * @param maskable
    *   batch x seq x ???
    * @param fill
    *   scalar
    */
  def sequenceMaskValidLength1D[S: Sc](
      maxLength: STen,
      maskable: Variable,
      fill: Double
  ) = {
    assert(maxLength.shape(0) == maskable.shape(0))
    assert(maxLength.shape.size == 1)
    val mask = (STen
      .arange_l(
        start = 0L,
        end = maskable.shape(2),
        step = 1L,
        tensorOptions = maskable.options
      )
      .unsqueeze(0) ge maxLength.unsqueeze(1)).unsqueeze(1)
    maskable.maskFill(
      lamp.autograd.const(mask),
      fill
    )

  }

  /** @param input
    *   batch x seq x ???
    * @param maxLength
    *   batch x seq OR batch , long
    * @return
    *   batch x seq x ???
    */
  def maskedSoftmax[S: Sc](
      input: Variable,
      maxLength: STen
  ) = {
    val maskedInput = sequenceMask(
      maxLength = maxLength,
      maskable = input,
      fill = Double.NegativeInfinity
    )
    maskedInput.logSoftMax(2).exp
  }

  /** Scaled dot product attention
    *
    * if maxLength is 2D: (batch,query,key) locations where
    * maxLength(batch,query) > key are ignored.
    *
    * if maxLength is 1D: (batch,query,key) locations where maxLength(batch) >
    * query are ignored
    *
    * See chapter 11.3.3 in d2l v1.0.0-beta0
    *
    * @param query
    *   batch x num queries x key dim
    * @param key
    *   batch x num k-v pairs x key dim
    * @param value
    *   batch x num k-v pairs x value dim
    * @param maxLength
    *   batch x num queries OR batch, type long
    * @return
    *   batch x num queries x value dim
    */
  def scaledDotProductAttention[S: Sc](
      query: Variable,
      keys: Variable,
      values: Variable,
      maxLength: Option[STen],
      dropout: Double,
      trainDropout: Boolean
  ) = {
    val d = query.shape(2)

    // batch x num queries x num k-v pairs
    val scores = query.bmm(keys.transpose(1, 2)) * (1d / math.sqrt(d.toDouble))
    // batch x num queries x num k-v pairs
    val weights =
      maxLength
        .fold(scores)(mx => maskedSoftmax(scores, mx))
        .dropout(dropout, trainDropout)

    weights.bmm(values)

  }

  /** Linearized dot product attention https://arxiv.org/pdf/2006.16236.pdf
    *
    * replaces exp(a dot b) with f(a) dot f(b) where f is any elementwise
    * function, in the paper f(x) = elu(x)+1 here f(x) = swish1(x)+1 due to this
    * decomposition a more efficient configuration of the chained matrix
    * multiplication may be used: (Q Kt) V = Q (Kt V)
    *
    * applies masking according to maskedSoftmax
    *
    * @param query
    *   batch x num queries x key dim
    * @param key
    *   batch x num k-v pairs x key dim
    * @param value
    *   batch x num k-v pairs x value dim
    * @param maxLength
    *   batch x num queries OR batch , type long
    * @return
    *   batch x num queries x value dim
    */
  def linearizedAttention[S: Sc](
      query: Variable,
      keys: Variable,
      values: Variable,
      maxLength: Option[STen],
      dropout: Double,
      trainDropout: Boolean
  ) = {

    val qF = (query.swish1 + 1)

    val maskable = (keys.swish1 + 1).dropout(dropout, trainDropout)

    // zero out some keys either due to padding or dropout
    val kF = maxLength.fold(maskable)(mx =>
      sequenceMask(
        maxLength = mx,
        maskable = maskable,
        fill = 0d
      )
    )

    val tmp1 = kF.transpose(1, 2).bmm(values)
    val tmp2 = kF.sum(dim = List(1), keepDim = true).transpose(1, 2)
    val enumerator = qF.bmm(tmp1)
    val denom = qF.bmm(tmp2)

    enumerator / (denom + 1e-5)

  }

  /** Multi-head scaled dot product attention
    *
    * See chapter 11.5 in d2l v1.0.0-beta0
    *
    * Attention masking is implemented similarly to chapter 11.3.2.1 in d2l.ai
    * v1.0.0-beta0. It supports unmasked attention, attention on variable length
    * input, and left-to-right attention.
    *
    * @param query
    *   batch x num queries x dq
    * @param key
    *   batch x num k-v pairs x dk
    * @param value
    *   batch x num k-v pairs x dv
    * @param maxLength
    *   batch x num queries OR batch , type long
    * @param wQuery
    *   dq x hidden
    * @param wKeys
    *   dk x hidden
    * @param wValues
    *   dv x hidden
    * @param wOutput
    *   hidden x po
    * @param numHeads
    *   number of output heads, must be divisible by hidden
    * @param linearized
    *   if true uses linearized attention. if false used scaled dot product
    *   attention
    * @return
    *   batch x num queries x po
    */
  def multiheadAttention[S: Sc](
      query: Variable,
      keys: Variable,
      values: Variable,
      maxLength: Option[STen],
      dropout: Double,
      trainDropout: Boolean,
      wQuery: Variable,
      wKeys: Variable,
      wValues: Variable,
      wOutput: Variable,
      numHeads: Int,
      linearized: Boolean,
      causalMask: Boolean
  ) = {

    def mm1(a: Variable, b: Variable) = {
      val shape = a.shape
      a.reshape(List(-1, shape.last)).mm(b).view(shape.dropRight(1) :+ -1L)
    }

    // in a x b x c
    // out a * h x b x c/h
    def transposeIn(x: Variable, h: Int) = {
      val shape1 = x.shape
      // batch x shape(1) x h x hidden/h
      val viewed = x.view(List(shape1(0), shape1(1), h, -1))
      // batch x h x shape(1) x hidden/h
      val transposed = viewed.transpose(1, 2)
      val shape2 = transposed.shape
      transposed.reshape(List(-1, shape2(2), shape2(3)))

    }

    // in : a * h x b x c
    // out : a x b x c * h
    def transposeOut(x: Variable, h: Int) = {
      val shape = x.shape
      val viewed = x.view(List(-1, h, shape(1), shape(2)))

      val transposed = viewed.transpose(1, 2)
      val shape2 = transposed.shape
      transposed.reshape(List(shape2(0), shape2(1), -1))
    }

    // batch x num queries x hidden
    val q1 = mm1(query, wQuery)
    // batch x num k-v x hidden
    val k1 = mm1(keys, wKeys)
    // batch x num k-v x hidden
    val v1 = mm1(values, wValues)

    val isCuda = q1.value.isCuda
    val nQ = q1.shape(1)
    val nK = k1.shape(1)
    val nV = v1.shape(1)
    val nB = q1.shape(0)
    val aligned =
      nQ % 8 == 0 && nK % 8 == 0 && nV % 8 == 0

    val useEfficientAttentionKernel =
      isCuda && aligned && nQ == nK && !linearized && (causalMask || maxLength.isEmpty) && (dropout == 0d || !trainDropout)

    val attention =
      if (useEfficientAttentionKernel)
        new ScaledDotProductAttention(
          implicitly[Scope],
          q1.view(List(nB, nQ, numHeads, -1)),
          k1.view(List(nB, nQ, numHeads, -1)),
          v1.view(List(nB, nQ, numHeads, -1)),
          causalMask
        ).value
          .flatten(2, 3)
      else {

        // (batch * numHeads) x num queries x hidden/numHeads
        val q1t: Variable = transposeIn(q1, numHeads)
        // (batch * numHeads) x num k-v x hidden/numHeads
        val k1t: Variable = transposeIn(k1, numHeads)
        // (batch * numHeads) x num k-v x hidden/numHeads
        val v1t: Variable = transposeIn(v1, numHeads)

        // (batch * numHeads) x num queries OR (batch * numHeads)
        val maxLengthRepated = if (causalMask && maxLength.isEmpty) {
          val single = STen.arange_l(1, nQ + 1, 1, q1t.options).unsqueeze(0)
          Some(single.repeat(List(nB * numHeads, 1)))

        } else maxLength.map(_.repeat(List(numHeads, 1)))

        // (batch * h) x num queries x hidden/h
        val output =
          if (linearized)
            linearizedAttention(
              q1t,
              k1t,
              v1t,
              maxLengthRepated,
              dropout,
              trainDropout
            )
          else
            scaledDotProductAttention(
              q1t,
              k1t,
              v1t,
              maxLengthRepated,
              dropout,
              trainDropout
            )

        // batch x num queries x hidden
        val outputConcat: Variable = transposeOut(output, numHeads)
        outputConcat
      }

    // batch x num queries x hidden
    mm1(attention, wOutput)

  }

}

object PositionalEmbedding {

  /** The trigonometric position encoding from https://arxiv.org/abs/1706.03762
    *
    * @param sequenceLength
    * @param dimension
    *   output dimension of the encoding
    * @param device
    * @param precision
    */
  def vaswani[S: Sc](
      sequenceLength: Int,
      dimension: Int,
      device: Device,
      precision: FloatingPointPrecision
  ) = {
    val m = Array.ofDim[Double](sequenceLength * dimension)
    var i = 0
    var j = 0
    val N = dimension / 2
    while (i < sequenceLength) {
      while (j < N) {
        val v1 = math.sin(i / math.pow(10000d, (2d * j) / dimension))
        val v2 = math.cos(i / math.pow(10000d, (2d * j) / dimension))
        m(i * dimension + 2 * j) = v1
        if (2 * j + 1 < dimension) {
          m(i * dimension + 2 * j + 1) = v2
        }
        j += 1
      }
      j = 0
      i += 1
    }
    STen.fromDoubleArray(m, List(sequenceLength, dimension), device, precision)
  }

  /** Linearly decomposed sequence distance encoding
    *
    * @param maxDistance
    * @param dimension
    *   output dimension
    * @return
    *   the first `dimension` left singular vectors of the row normalized p
    *   where p(i,j) = min(maxDist,abs(i-j))
    */
  def simpleSequence(
      sequenceLength: Int,
      dimension: Int,
      maxDistance: Int,
      device: Device,
      precision: FloatingPointPrecision
  )(implicit
      scope: Scope
  ) = {
    val m = Array.ofDim[Double](sequenceLength * sequenceLength)
    var i = 0
    var j = 0
    while (i < sequenceLength) {
      while (j < sequenceLength) {
        val v = math.min(maxDistance, math.abs(j - i))
        m(i * sequenceLength + j) = v
        j += 1
      }
      j = 0
      i += 1
    }
    lamp.Scope { implicit scope =>
      val t = STen.fromDoubleArray(
        m,
        List(sequenceLength, sequenceLength),
        device,
        precision
      )
      val len = (t * t).sum(dim = 1, keepDim = false).sqrt
      val normed = t / len.view(-1, 1)
      val (u, s, _) = normed.svd()
      val ut = u.t
      val m3 = ut.slice(1, 0, dimension, 1) * (s.slice(0, 0, dimension, 1).sqrt)
      m3
    }
  }
}

/** A module with positional and token embeddings
  *
  * Token embeddings are lookup embeddings. Positional embeddings are supplied
  * as a constant. They are supposed to come from a fixed unlearned derivation
  * of the positions.
  *
  * Token and positional embeddings are summed.
  *
  * Gradients are not computed for `positionalEmbedding`
  */
case class TransformerEmbedding(
    embedding: lamp.nn.Embedding,
    addPositionalEmbedding: Boolean,
    positionalEmbedding: Constant
) extends GenericModule[Variable, Variable] {
  def state =
    List(
      positionalEmbedding -> TransformerEmbedding.Embedding
    ) ++ embedding.state
  def forward[S: Sc](x: Variable): Variable = {
    val embedded = embedding.forward(x)
    val viewed = positionalEmbedding.view(1L +: positionalEmbedding.shape)
    val withPos =
      if (addPositionalEmbedding) embedded + viewed
      else
        embedded.cat(
          const(viewed.value.repeat(List(embedded.shape(0), 1L, 1L))),
          2
        )
    withPos
  }
}
object TransformerEmbedding {
  case object Embedding extends LeafTag
  implicit val trainingMode: TrainingMode[TransformerEmbedding] =
    TrainingMode.make[TransformerEmbedding](
      m => m.copy(embedding = m.embedding.asEval),
      m => m.copy(embedding = m.embedding.asTraining)
    )
  implicit val load: Load[TransformerEmbedding] =
    Load.make[TransformerEmbedding] { m => tensors =>
      m.positionalEmbedding.value.copyFrom(tensors(0))
      m.embedding.load(tensors.drop(1))

    }

}
