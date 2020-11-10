package lamp.nn

import lamp.autograd.Variable
import lamp.autograd.ConcatenateAddNewDim
import lamp.Sc
import lamp.scope

object Attention {

  /**
    * @param tokens seq x batch (long)
    * @param maskable batch x seq
    * @param maskedToken
    * @param fill
    * @return batch x seq where (seq,batch,:) is set to fill if tokens(seq,batch)== maskedToken
    */
  def sequenceMask[S: Sc](
      tokens: Variable,
      maskable: Variable,
      maskedToken: Long,
      fill: Double
  ) = {
    assert(tokens.shape(0) == maskable.shape(1))
    assert(tokens.shape(1) == maskable.shape(0))
    assert(tokens.shape.size == 2)
    assert(maskable.shape.size == 2)
    val tokensT = tokens.transpose(0, 1)
    val mask = tokensT.makeBooleanMask(maskedToken)
    maskable.maskFill(mask, fill)

  }

  /** Dot product attention
    * @param query  batch x d
    * @param key num keys x batch x d
    * @return  batch x d
    */
  def dotProductAttention[S: Sc](
      query: Variable,
      keyvalue: Variable,
      tokens: Variable,
      padToken: Long
  ) = {
    val d = query.shape(1)
    val k = keyvalue.shape(0)
    val batch = query.shape(0)

    // batch x d x 1
    val queryT = query.view(List(batch.toInt, d.toInt, 1))
    // batch x keys x d
    val keyT = keyvalue.transpose(0, 1)
    // batch x 1 x keys
    val a = keyT
      .bmm(queryT)
      .*(1 / math.sqrt(d.toDouble))
      .view(List(batch.toInt, -1))

    val aMasked = sequenceMask(
      tokens = tokens,
      maskable = a,
      maskedToken = padToken,
      fill = -100000d
    )

    // batch x 1 x keys
    val sm = aMasked.view(List(batch.toInt, 1, k.toInt)).logSoftMax(2).exp
    // batch x 1 x d2
    val output = sm.bmm(keyT)
    // 1 x batch x d2
    output.view(List(batch.toInt, -1))
  }

  def forward[T, M <: StatefulModule[Variable, Variable, T], S: Sc](
      decoder: M with StatefulModule[Variable, Variable, T],
      x: Variable,
      keyValue: Variable,
      state: T,
      tokens: Variable,
      padToken: Long
  )(stateToKey: T => Variable) = {
    val timesteps = x.shape.head
    val outputs = scala.collection.mutable.ArrayBuffer[Variable]()
    val lastHidden =
      (0 until timesteps.toInt).foldLeft(state) { (h, t) =>
        val xt = x.select(0, t)
        val context =
          Attention
            .dotProductAttention(
              query = stateToKey(h),
              keyvalue = keyValue,
              tokens = tokens,
              padToken = padToken
            )

        val catted = xt.cat(context, dim = 1)
        val viewed = catted.view((1L :: catted.shape).map(_.toInt))
        val (output, nextHidden) = decoder.forward((viewed, h))

        outputs.append(output.select(0, 0))
        nextHidden
      }
    val r = ConcatenateAddNewDim(scope, outputs).value
    (r, lastHidden)

  }
}

case class AttentionDecoder[T, M <: StatefulModule[Variable, Variable, T], M0 <: Module](
    decoder: M with StatefulModule[Variable, Variable, T],
    embedding: M0 with Module,
    stateToKey: T => Variable,
    keyValue: Variable,
    tokens: Variable,
    padToken: Long
) extends StatefulModule[Variable, Variable, T] {

  override def state: Seq[(Variable, PTag)] =
    decoder.state ++ embedding.state

  def forward[S: Sc](x: (Variable, T)) = {
    val (input, state) = x
    forward1(input, state)
  }
  def forward1[S: Sc](x: Variable, state: T) =
    Attention.forward(
      decoder,
      embedding.forward(x),
      keyValue,
      state,
      tokens,
      padToken
    )(
      stateToKey
    )

}
