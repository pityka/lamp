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

/** TransformerEncoder module
  *
  * Input is `(data, tokens)` where
  * `data` is (batch, num tokens, in dimension), double tensor
  * `tokens` is (batch,num tokens) long tensor.
  *
  * Output is (bach, num tokens, out dimension)
  *
  * The sole purpose of `tokens` is to carry over the padding
  */
case class TransformerEncoder(
    blocks: Seq[TransformerEncoderBlock]
) extends GenericModule[(Variable, STen), Variable] {
  def state = blocks.map(_.state).foldLeft(List.empty[(Constant, PTag)])(_ ++ _)
  def forward[S: Sc](x: (Variable, STen)): Variable = {
    val (input, tokens) = x
    blocks.foldLeft(input) { (a, block) => block.forward((a, tokens)) }
  }
}
object TransformerEncoder {
  implicit val trainingMode = TrainingMode
    .make[TransformerEncoder](
      m => m.copy(blocks = m.blocks.map(_.asEval)),
      m => m.copy(blocks = m.blocks.map(_.asTraining))
    )
  implicit val load = Load.make[TransformerEncoder] { m => tensors =>
    m.blocks.foldLeft((List[Unit](), tensors)) {
      case ((acc, params), member) =>
        val numParam = member.state.size
        val loaded = member.load(params.take(numParam))
        (acc.:+(loaded), params.drop(numParam))

    }

  }

  /**
    * Factory for the encoder module of transformer
    * Does *not* include embedding and positional encoding
    *
    * Input is `(data, tokens)` where
    * `data` is (batch, num tokens, in dimension), double tensor
    * `tokens` is (batch,num tokens) long tensor.
    *
    * The sole purpose of `tokens` is to carry over the padding
    *
    * @param numBlocks number of transformer blocks to create
    * @param in input dimension
    * @param attentionHiddenPerHeadDim size of hidden attention dimension of each attention head
    * @param attentionNumHeads number of attention heads
    * @param mlpHiddenDim size of hidden dimension of the two layer perceptron
    * @param out output dimension
    * @param dropout dropout rate
    * @param padToken pad token, (batch, seq) positions where `tokens` == `padToken` are ignored
    * @param tOpt tensor options
    * @return a module
    */
  def apply[S: Sc](
      numBlocks: Int,
      in: Int,
      attentionHiddenPerHeadDim: Int,
      attentionNumHeads: Int,
      mlpHiddenDim: Int,
      dropout: Double,
      padToken: Long,
      tOpt: STenOptions,
      linearized: Boolean
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
          padToken = padToken,
          tOpt = tOpt,
          linearized = linearized
        )
      )
    )
}

/** A single block of the transformer encoder as defined in Fig 10.7.1 in d2l v0.16
  */
case class TransformerEncoderBlock(
    attention: MultiheadAttention,
    layerNorm1: LayerNorm,
    layerNorm2: LayerNorm,
    w1: Constant,
    b1: Constant,
    w2: Constant,
    b2: Constant,
    dropout: Double,
    train: Boolean
) extends GenericModule[(Variable, STen), Variable] {

  def state =
    attention.state ++ layerNorm1.state ++ layerNorm2.state ++ Seq(
      w1 -> TransformerEncoderBlock.Weights1,
      w2 -> TransformerEncoderBlock.Weights2,
      b1 -> TransformerEncoderBlock.Bias1,
      b2 -> TransformerEncoderBlock.Bias2
    )

  def forward[S: Sc](x: (Variable, STen)): Variable = {

    def mm1(a: Variable, b: Variable) = {
      val shape = a.shape
      a.view(List(-1, shape.last)).mm(b).view(shape.dropRight(1) :+ -1L)
    }

    val (input, tokens) = x
    val a1 = attention.forward((input, input, input, tokens))
    val a2 = layerNorm1(a1.dropout(dropout, train) + input)
    val a3 = mm1((mm1(a2, w1) + b1).relu, w2) + b2

    val a4 = layerNorm2(a3.dropout(dropout, train) + a3)
    a4
  }

}

object TransformerEncoderBlock {

  def apply[S: Sc](
      in: Int,
      attentionHiddenPerHeadDim: Int,
      attentionNumHeads: Int,
      mlpHiddenDim: Int,
      out: Int,
      dropout: Double,
      padToken: Long,
      tOpt: STenOptions,
      linearized: Boolean
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
        padToken = padToken,
        tOpt = tOpt,
        linearized = linearized
      ),
      layerNorm1 = LayerNorm(List(2), tOpt),
      layerNorm2 = LayerNorm(List(2), tOpt),
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

    }
}

/**
  * Multi-head scaled dot product attention module
  *
  * Input: (query,key,value,tokens) where
  *  query: batch x num queries x query dim
  *  key: batch x num k-v x key dim
  *  value: batch x num k-v x key value
  *  tokens: batch x num queries, long type
  *
  * Tokens is used to carry over padding information and ignore the padding
  *
  */
case class MultiheadAttention(
    wQ: Constant,
    wK: Constant,
    wV: Constant,
    wO: Constant,
    dropout: Double,
    train: Boolean,
    numHeads: Int,
    padToken: Long,
    linearized: Boolean
) extends GenericModule[(Variable, Variable, Variable, STen), Variable] {

  override val state = List(
    wQ -> MultiheadAttention.WeightsQ,
    wK -> MultiheadAttention.WeightsK,
    wV -> MultiheadAttention.WeightsV,
    wO -> MultiheadAttention.WeightsO
  )

  override def forward[S: Sc](
      x: (Variable, Variable, Variable, STen)
  ): Variable = {
    val (q, k, v, tokens) = x

    MultiheadAttention.multiheadAttention(
      query = q,
      keys = k,
      values = v,
      tokens = tokens,
      padToken = padToken,
      dropout = dropout,
      trainDropout = train,
      wQuery = wQ,
      wKeys = wK,
      wValues = wV,
      wOutput = wO,
      numHeads = numHeads,
      linearized = linearized
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
      padToken: Long,
      tOpt: STenOptions,
      linearized: Boolean
  ): MultiheadAttention = MultiheadAttention(
    wQ = initLinear(dQ, hiddenPerHead * numHeads, tOpt),
    wK = initLinear(dK, hiddenPerHead * numHeads, tOpt),
    wV = initLinear(dV, hiddenPerHead * numHeads, tOpt),
    wO = initLinear(hiddenPerHead * numHeads, out, tOpt),
    dropout = dropout,
    train = true,
    numHeads = numHeads,
    padToken = padToken,
    linearized = linearized
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

  /**
    * @param tokens batch x seq , type long
    * @param maskable batch x seq x ???
    * @param pad
    * @param fill
    * @return batch x seq x ??? where (seq,batch,:) is set to fill if tokens(seq,batch)== maskedToken
    */
  def sequenceMask[S: Sc](
      tokens: STen,
      maskable: Variable,
      pad: Long,
      fill: Double
  ) = {
    assert(tokens.shape(1) == maskable.shape(1))
    assert(tokens.shape(0) == maskable.shape(0))
    assert(tokens.shape.size == 2)
    val mask = tokens.equ(pad)
    maskable.maskFill(
      lamp.autograd.const(mask.view(mask.shape :+ 1L: _*)),
      fill
    )

  }

  /**
    * @param input batch x seq x ???
    * @param mask scalar long
    * @param tokens batch x seq , long
    * @return batch x seq x ???
    */
  def maskedSoftmax[S: Sc](
      input: Variable,
      pad: Long,
      tokens: STen
  ) = {
    val maskedInput = sequenceMask(
      tokens = tokens,
      maskable = input,
      pad = pad,
      fill = -1000000d
    )
    maskedInput.logSoftMax(2).exp
  }

  /** Scaled dot product attention
    *
    * (batch,query) locations where tokens(batch,query) == pad are ignored
    *
    * @param query  batch x num queries x key dim
    * @param key batch x num k-v pairs x key dim
    * @param value batch x num k-v pairs x value dim
    * @param tokens batch x num queries , type long
    * @param pad scalar long
    * @return  batch x num queries x value dim
    */
  def scaledDotProductAttention[S: Sc](
      query: Variable,
      keys: Variable,
      values: Variable,
      tokens: STen,
      padToken: Long,
      dropout: Double,
      trainDropout: Boolean
  ) = {
    val d = query.shape(2)

    // batch x num queries x num k-v pairs
    val scores = query.bmm(keys.transpose(1, 2)) * (1d / math.sqrt(d.toDouble))
    // batch x num queries x num k-v pairs
    val weights =
      maskedSoftmax(scores, padToken, tokens).dropout(dropout, trainDropout)

    weights.bmm(values)

  }

  /** Linearized dot product attention
    *  https://arxiv.org/pdf/2006.16236.pdf
    *
    * replaces exp(a dot b) with f(a) dot f(b)
    * where f is any elementwise function,
    *  in the paper f(x) = elu(x)+1
    *  here f(x) = swish1(x)+1
    * due to this decomposition a more efficient configuration of the chained matrix multiplication
    * may be used: (Q Kt) V = Q (Kt V)
    *
    * (batch,query) locations where tokens(batch,query) == pad are ignored
    *
    * @param query  batch x num queries x key dim
    * @param key batch x num k-v pairs x key dim
    * @param value batch x num k-v pairs x value dim
    * @param tokens batch x num queries , type long
    * @param pad scalar long
    * @return  batch x num queries x value dim
    */
  def linearizedAttention[S: Sc](
      query: Variable,
      keys: Variable,
      values: Variable,
      tokens: STen,
      padToken: Long,
      dropout: Double,
      trainDropout: Boolean
  ) = {

    val qF = (query.swish1 + 1)

    // zero out some keys either due to padding or dropout
    val kF = sequenceMask(
      tokens,
      (keys.swish1 + 1).dropout(dropout, trainDropout),
      padToken,
      0d
    )

    val tmp1 = kF.transpose(1, 2).bmm(values)
    val tmp2 = kF.sum(dim = List(1), keepDim = true).transpose(1, 2)
    val enumerator = qF.bmm(tmp1)
    val denom = qF.bmm(tmp2)

    enumerator / (denom + 1e-5)

  }

  /** Multi-head scaled dot product attention
    *
    * (batch,query) locations where tokens(batch,query) == pad are ignored
    *
    * @param query  batch x num queries x dq
    * @param key batch x num k-v pairs x dk
    * @param value batch x num k-v pairs x dv
    * @param tokens batch x num queries , type long
    * @param pad scalar long
    * @param wQuery dq x hidden
    * @param wKeys dk x hidden
    * @param wValues  dv x hidden
    * @param wOutput  hidden  x po
    * @param numHeads number of output heads, must be divisible by hidden
    * @return  batch x num queries x po
    */
  def multiheadAttention[S: Sc](
      query: Variable,
      keys: Variable,
      values: Variable,
      tokens: STen,
      padToken: Long,
      dropout: Double,
      trainDropout: Boolean,
      wQuery: Variable,
      wKeys: Variable,
      wValues: Variable,
      wOutput: Variable,
      numHeads: Int,
      linearized: Boolean
  ) = {

    def mm1(a: Variable, b: Variable) = {
      val shape = a.shape
      a.view(List(-1, shape.last)).mm(b).view(shape.dropRight(1) :+ -1L)
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

    // (batch * numHeads) x num queries x hidden/numHeads
    val q1t: Variable = transposeIn(q1, numHeads)
    // (batch * numHeads) x num k-v x hidden/numHeads
    val k1t: Variable = transposeIn(k1, numHeads)
    // (batch * numHeads) x num k-v x hiddenhnumHeads
    val v1t: Variable = transposeIn(v1, numHeads)

    // (batch * numHeads) x num queries
    val tokensRepeated = tokens.repeat(List(numHeads, 1))

    // (batch * h) x num queries x hidden/h
    val output =
      if (linearized)
        linearizedAttention(
          q1t,
          k1t,
          v1t,
          tokensRepeated,
          padToken,
          dropout,
          trainDropout
        )
      else
        scaledDotProductAttention(
          q1t,
          k1t,
          v1t,
          tokensRepeated,
          padToken,
          dropout,
          trainDropout
        )

    // batch x num queries x hidden
    val outputConcat: Variable = transposeOut(output, numHeads)
    mm1(outputConcat, wOutput)

  }

}

object PositionalEmbedding {

  def vaswani[S: Sc](
      sequenceLength: Int,
      dimension: Int,
      device: Device,
      precision: FloatingPointPrecision
  ) = {
    val m = org.saddle.mat.zeros(sequenceLength, dimension)
    var i = 0
    var j = 0
    while (i < sequenceLength) {
      while (j < dimension / 2) {
        val v1 = math.sin(i / math.pow(10000d, (2d * j) / dimension))
        val v2 = math.cos(i / math.pow(10000d, (2d * j) / dimension))
        m.mutateSetCell(i, 2 * j, v1)
        if (2 * j + 1 < dimension) {
          m.mutateSetCell(i, 2 * j + 1, v2)
        }
        j += 1
      }
      j = 0
      i += 1
    }
    STen.fromMat(m, device, precision)
  }

  /**
    * p(i,j) = min(maxDist,abs(i-j))
    * returns the first `dimension` left singular vectors of the row normalized p
    */
  def simpleSequence(
      sequenceLength: Int,
      dimension: Int,
      maxDistance: Int,
      device: Device,
      precision: FloatingPointPrecision
  )(
      implicit scope: Scope
  ) = {
    val m = org.saddle.mat.zeros(sequenceLength, sequenceLength)
    var i = 0
    var j = 0
    while (i < sequenceLength) {
      while (j < sequenceLength) {
        val v = math.min(maxDistance, math.abs(j - i))
        m.mutateSetCell(i, j, v)
        j += 1
      }
      j = 0
      i += 1
    }
    lamp.Scope { implicit scope =>
      val t = STen.fromMat(m, device, precision)
      val len = (t * t).sum(dim = 1, keepDim = false).sqrt
      val normed = t / len.view(-1, 1)
      val (ut, s, _) = normed.svd
      val m3 = ut.slice(1, 0, dimension, 1) * (s.slice(0, 0, dimension, 1).sqrt)
      m3
    }
  }
}

/**
  * Gradients are not computed for `positionalEmbedding`
  */
case class TransformerEmbedding(
    embedding: lamp.nn.Embedding,
    addPositionalEmbedding: Boolean,
    positionalEmbedding: Constant
) extends GenericModule[Variable, (Variable, STen)] {
  def state =
    List(positionalEmbedding -> TransformerEmbedding.Embedding) ++ embedding.state
  def forward[S: Sc](x: Variable): (Variable, STen) = {
    val embedded = embedding.forward(x)
    val viewed = positionalEmbedding.view(1L +: positionalEmbedding.shape)
    val withPos =
      if (addPositionalEmbedding) embedded + viewed
      else
        embedded.cat(
          const(viewed.value.repeat(List(embedded.shape(0), 1L, 1L))),
          2
        )
    (withPos, x.value)
  }
}
object TransformerEmbedding {
  case object Embedding extends LeafTag
  implicit val trainingMode = TrainingMode.make[TransformerEmbedding](
    m => m.copy(embedding = m.embedding.asEval),
    m => m.copy(embedding = m.embedding.asTraining)
  )
  implicit val load = Load.make[TransformerEmbedding] { m => tensors =>
    m.positionalEmbedding.value.copyFrom(tensors(0))
    m.embedding.load(tensors.drop(1))

  }

}
