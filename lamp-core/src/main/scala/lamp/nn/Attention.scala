package lamp.nn

import lamp.autograd.Variable
import lamp.autograd.ConcatenateAddNewDim

object Attention {

  /** Dot product attention
    * @param query  batch x d
    * @param key num keys x batch x d
    * @param value num keys x batch x d2
    * @return  batch x d2
    */
  def dotProductAttention(
      query: Variable,
      key: Variable,
      value: Variable
  ) = {
    val d = query.shape(1)
    val batch = query.shape(0)
    // batch x 1 x d
    val queryT = query.view(List(batch.toInt, 1, d.toInt))
    // batch x d x keys
    val keyT = key.transpose(0, 1).transpose(1, 2)
    // batch x 1 x keys
    val a = queryT.bmm(keyT).*(math.sqrt(d.toDouble))
    // batch x 1 x keys
    val sm = a.logSoftMax(2).exp
    // batch x 1 x d2
    val output = sm.bmm(value.transpose(0, 1))
    // 1 x batch x d2
    val outputT = output.transpose(0, 1)
    output.view(List(batch.toInt, -1))
  }

  def forward[T, M <: StatefulModule[Variable, Variable, T], M2 <: Module](
      decoder: M with StatefulModule[Variable, Variable, T],
      contextModule: M2 with Module,
      x: Variable,
      keyValue: Variable,
      state: T
  )(stateToKey: T => Variable) = {
    val timesteps = x.shape.head
    val batchSize = x.shape(1)
    val outputs = scala.collection.mutable.ArrayBuffer[Variable]()
    val lastHidden =
      (0 until timesteps.toInt).foldLeft(state) { (h, t) =>
        val xt = x.select(0, t)
        val context =
          contextModule.forward(
            Attention.dotProductAttention(
              query = stateToKey(h),
              key = keyValue,
              value = keyValue
            )
          )
        val catted = xt.cat(context, dim = 1)
        val viewed = catted.view((1L :: catted.shape).map(_.toInt))
        val (output, nextHidden) = decoder.forward((viewed, h))

        outputs.append(output.select(0, 0))
        nextHidden
      }
    val r = ConcatenateAddNewDim(outputs).value
    (r, lastHidden)

  }
}

case class AttentionDecoder[T, M <: StatefulModule[Variable, Variable, T], M0 <: Module, M2 <: Module](
    decoder: M with StatefulModule[Variable, Variable, T],
    embedding: M0 with Module,
    contextModule: M2 with Module,
    stateToKey: T => Variable,
    keyValue: Variable
) extends StatefulModule[Variable, Variable, T] {

  override def state: Seq[(Variable, PTag)] =
    decoder.state ++ contextModule.state

  def forward(x: (Variable, T)) = {
    val (input, state) = x
    forward1(input, state)
  }
  def forward1(x: Variable, state: T) =
    Attention.forward(
      decoder,
      contextModule,
      embedding.forward(x),
      keyValue,
      state
    )(
      stateToKey
    )

}
