package lamp.nn

import lamp.autograd.{const, Variable}
import lamp._
import lamp.autograd.Constant

case class GraphAttention(
    wNodeKey1: Constant,
    wNodeKey2: Constant,
    wEdgeKey: Constant,
    wNodeValue: Constant,
    wAttention: Option[Constant],
    nonLinearity: Boolean,
    dropout: Dropout,
    numHeads: Int
) extends GenericModule[GraphAttention.Graph, GraphAttention.Graph] {

  override def forward[S: Sc](x: GraphAttention.Graph): GraphAttention.Graph = {
    val activation = GraphAttention.multiheadGraphAttention(
      nodeFeatures = x.nodeFeatures,
      edgeFeatures = x.edgeFeatures,
      edgeI = x.edgeI,
      edgeJ = x.edgeJ,
      wNodeKey1 = wNodeKey1,
      wNodeKey2 = wNodeKey2,
      wEdgeKey = wEdgeKey,
      wNodeValue = wNodeValue,
      wAttention = wAttention,
      numHeads = numHeads
    )

    val nextNodeFeatures =
      if (nonLinearity) dropout.forward(activation.swish1)
      else activation

    val residual =
      if (nextNodeFeatures.shape == x.nodeFeatures.shape)
        x.nodeFeatures + nextNodeFeatures
      else nextNodeFeatures

    x.copy(nodeFeatures = residual)
  }

  def state: Seq[(Constant, PTag)] =
    dropout.state ++ (List(
      wNodeKey1,
      wNodeKey2,
      wEdgeKey,
      wNodeValue
    ) ++ wAttention.toList).map(
      _ -> GraphAttention.Weights
    )
}

object GraphAttention {

  def apply[S: Sc](
      nodeDim: Int,
      edgeDim: Int,
      attentionKeyHiddenDimPerHead: Int,
      attentionNumHeads: Int,
      valueDimPerHead: Int,
      dropout: Double,
      tOpt: STenOptions,
      dotProductAttention: Boolean,
      nonLinearity: Boolean
  ): GraphAttention = GraphAttention(
    wNodeKey1 = initLinear(
      nodeDim,
      attentionKeyHiddenDimPerHead * attentionNumHeads,
      tOpt
    ),
    wNodeKey2 = initLinear(
      nodeDim,
      attentionKeyHiddenDimPerHead * attentionNumHeads,
      tOpt
    ),
    wEdgeKey =
      if (dotProductAttention)
        initLinear(
          edgeDim,
          attentionNumHeads,
          tOpt
        )
      else
        initLinear(
          edgeDim,
          attentionKeyHiddenDimPerHead * attentionNumHeads,
          tOpt
        ),
    wNodeValue = initLinear(nodeDim, valueDimPerHead * attentionNumHeads, tOpt),
    wAttention =
      if (dotProductAttention) None
      else
        Some(
          initLinear(
            attentionKeyHiddenDimPerHead * 3,
            attentionNumHeads,
            tOpt
          )
        ),
    nonLinearity = nonLinearity,
    dropout = Dropout(dropout, true),
    numHeads = attentionNumHeads
  )

  case object Weights extends LeafTag

  case class Graph(
      nodeFeatures: Variable,
      edgeFeatures: Variable,
      edgeI: STen,
      edgeJ: STen
  )

  /** Graph Attention Network https://arxiv.org/pdf/1710.10903.pdf Non-linearity
    * in eq 4 and dropout is not applied to the final vertex activations
    *
    * Needs self edges to be already present in the graph
    *
    * @return
    *   next node representation (without relu, dropout) and a tensor with the
    *   original node and edge features ligned up like [N_i, N_j, E_ij]
    */
  def multiheadGraphAttention[S: Sc](
      nodeFeatures: Variable,
      edgeFeatures: Variable,
      edgeI: STen,
      edgeJ: STen,
      wNodeKey1: Variable,
      wNodeKey2: Variable,
      wEdgeKey: Variable,
      wNodeValue: Variable,
      wAttention: Option[Variable],
      numHeads: Int
  ) = {

    def mm(a: Variable, b: Variable) = 
      a.mm(b).view(List(a.shape(0),numHeads,b.shape(1)/numHeads))
    

    val nodeKey1 = mm(nodeFeatures, wNodeKey1)
    val nodeKey2 = mm(nodeFeatures, wNodeKey2)
    val edgeKey = mm(edgeFeatures, wEdgeKey)
    val nodeValue = mm(nodeFeatures, wNodeValue)

    val activations = wAttention match {
      case Some(wAttention) =>
        val ninjeij = {
          Variable.cat(
            List(
              nodeKey1
                .indexSelect(dim = 0, const(edgeI)),
              nodeKey2
                .indexSelect(dim = 0, const(edgeJ)),
              edgeKey
            ),
            dim = 2
          )
        }

        val K = ninjeij.shape(2)
        (ninjeij
          .transpose(0, 1) bmm wAttention
          .view(List(K, numHeads, 1))
          .transpose(0, 1)).tanh.transpose(0, 1).view(List(-1, numHeads))

      case None =>
        val ni =
          nodeKey1.indexSelect(dim = 0, const(edgeI))
        val nj =
          nodeKey2.indexSelect(dim = 0, const(edgeJ))
        val prod = ((ni * nj) * (1d / math.sqrt(ni.shape(1).toDouble)))
        val dot = prod
          .sum(dim = List(2), keepDim = true)
        dot + edgeKey.reshape(List(-1, numHeads, 1))

    }
    val c = const(activations.value.max)
    val e = (activations - c).exp
    val lse = e.indexAdd(const(edgeJ), 0, nodeFeatures.shape(0)).log + c
    val lseBroadCast = lse.indexSelect(dim = 0, const(edgeJ))
    val logsoftmax = activations - lseBroadCast
    val a = logsoftmax.exp.view(List(-1, numHeads, 1))

    assert(
      nodeValue.shape(1) % numHeads == 0,
      s"wNodeValue and numHeads size do not align ${wNodeValue
        .shape(1)} $numHeads"
    )

    val h = {
      val nodeValueScatter = nodeValue
        .indexSelect(dim = 0, const(edgeI))

      (a * nodeValueScatter)
        .reshape(
          List(-1, nodeValueScatter.shape(1) * nodeValueScatter.shape(2))
        )
        .indexAdd(const(edgeJ), 0, nodeFeatures.shape(0))

    }

    h
  }

  implicit val tr: TrainingMode[GraphAttention] = TrainingMode
    .make[GraphAttention](
      m => m.copy(dropout = m.dropout.asEval),
      m => m.copy(dropout = m.dropout.asTraining)
    )
  implicit val load = Load.make[GraphAttention] { m => parameters =>
    m.wNodeKey1.value.copyFrom(parameters(0))
    m.wNodeKey2.value.copyFrom(parameters(1))
    m.wEdgeKey.value.copyFrom(parameters(2))
    m.wNodeValue.value.copyFrom(parameters(3))
    m.wAttention.foreach(_.value.copyFrom(parameters(4)))

  }

}
