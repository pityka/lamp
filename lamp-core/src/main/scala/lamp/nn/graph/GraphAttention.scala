package lamp.nn.graph

import lamp.autograd.{const, Variable}
import lamp._
import lamp.nn._
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
) extends GenericModule[Graph, Graph] {

  override def forward[S: Sc,F:FW](x: Graph): Graph = {
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

  /** Graph Attention Network https://arxiv.org/pdf/1710.10903.pdf Non-linearity
    * in eq 4 and dropout is not applied to the final vertex activations
    *
    * Needs self edges to be already present in the graph
    *
    * @return
    *   next node representation (without relu, dropout) and a tensor with the
    *   original node and edge features ligned up like [N_i, N_j, E_ij]
    */
  def multiheadGraphAttention[S: Sc,F:FW](
      nodeFeatures: Variable,
      edgeFeatures: Variable,
      edgeI: STen,
      edgeJ: STen,
      wNodeKey1: Constant,
      wNodeKey2: Constant,
      wEdgeKey: Constant,
      wNodeValue: Constant,
      wAttention: Option[Constant],
      numHeads: Int
  ) = {

    def mm(a: Variable, b: Variable) =
      a.mm(b).view(List(a.shape.apply(0), numHeads, b.shape.apply(1) / numHeads))

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
                .indexSelect(dim = 0, (edgeI)),
              nodeKey2
                .indexSelect(dim = 0, (edgeJ)),
              edgeKey
            ),
            dim = 2
          )
        }

        val K = ninjeij.shape.apply(2)
        (ninjeij
          .transpose(0, 1) bmm wAttention
          .view(List(K, numHeads, 1))
          .transpose(0, 1)).tanh.transpose(0, 1).view(List(-1, numHeads))

      case None =>
        val ni =
          nodeKey1.indexSelect(dim = 0, (edgeI))
        val nj =
          nodeKey2.indexSelect(dim = 0, (edgeJ))
        val prod = ((ni * nj) * (1d / math.sqrt(ni.shape.apply(1).toDouble)))
        val dot = prod
          .sum(dim = List(2), keepDim = true)
        dot + edgeKey.reshape(List(-1, numHeads, 1))

    }
    val c = const(activations.forward.max)
    val e = (activations - c).exp
    val lse = e.indexAdd((edgeJ), 0, nodeFeatures.shape.apply(0)).log + c
    val lseBroadCast = lse.indexSelect(dim = 0, (edgeJ))
    val logsoftmax = activations - lseBroadCast
    val a = logsoftmax.exp.view(List(-1, numHeads, 1))

    assert(
      nodeValue.shape.apply(1) % numHeads == 0,
      s"wNodeValue and numHeads size do not align ${wNodeValue
        .shape.apply(1)} $numHeads"
    )

    val h = {
      val nodeValueScatter = nodeValue
        .indexSelect(dim = 0, (edgeI))

      (a * nodeValueScatter)
        .reshape(
          List(-1, nodeValueScatter.shape.apply(1) * nodeValueScatter.shape.apply(2))
        )
        .indexAdd((edgeJ), 0, nodeFeatures.shape.apply(0))

    }

    h
  }

  implicit val tr: TrainingMode[GraphAttention] = TrainingMode
    .make[GraphAttention](
      m => m.copy(dropout = m.dropout.asEval),
      m => m.copy(dropout = m.dropout.asTraining)
    )
  implicit val load : Load[GraphAttention] = Load.make[GraphAttention] { m => parameters =>
    
    m.wNodeKey1.constantValue.copyFrom(parameters(0))
    m.wNodeKey2.constantValue.copyFrom(parameters(1))
    m.wEdgeKey.constantValue.copyFrom(parameters(2))
    m.wNodeValue.constantValue.copyFrom(parameters(3))
    m.wAttention.foreach(_.constantValue.copyFrom(parameters(4)))

  }

}
