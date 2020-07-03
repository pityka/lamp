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
    val queryT = query.view(List(batch.toInt, 1, d.toInt))
    val keyT = key.transpose(0, 1).transpose(1, 2)
    // batch x 1 x keys
    val a = queryT.bmm(keyT).*(math.sqrt(d.toDouble))
    // batch x 1 x keys
    val sm = a.logSoftMax(2)
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

// case class AttentionDecoder[T, M <: StatefulModule[Variable, Variable, T]](
//     module: M with StatefulModule[Variable, Variable, T]
// ) extends StatefulModule[(Variable, Variable), Variable, T] {

//   override def state: Seq[(Variable, PTag)] = module.state

//   def forward(a: ((Variable, Variable), T)) = forward1(a._1._1, a._1._2, a._2)

// }
