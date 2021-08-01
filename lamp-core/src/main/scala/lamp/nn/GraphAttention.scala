package lamp.nn

import lamp.autograd.{const, Variable}
import lamp._
import lamp.autograd.Constant

case class GraphAttention(
    wNodeKey1: Constant,
    wNodeKey2: Constant,
    wEdgeKey: Constant,
    wNodeValue: Constant,
    wAttention: Constant,
    nonLinearity: Boolean,
    dropout: Dropout
) extends GenericModule[GraphAttention.Graph, GraphAttention.Graph] {

  override def forward[S: Sc](x: GraphAttention.Graph): GraphAttention.Graph = {
    val (activation, _) = GraphAttention.multiheadGraphAttention(
      nodeFeatures = x.nodeFeatures,
      edgeFeatures = x.edgeFeatures,
      edgeI = x.edgeI,
      edgeJ = x.edgeJ,
      wNodeKey1 = wNodeKey1,
      wNodeKey2 = wNodeKey2,
      wEdgeKey = wEdgeKey,
      wNodeValue = wNodeValue,
      wAttention = wAttention
    )

    val nextNodeFeatures =
      if (nonLinearity) dropout.forward(activation.swish1)
      else activation

    val residual =
      if (nextNodeFeatures.shape == x.nodeFeatures.shape)
        x.nodeFeatures + nextNodeFeatures
      else x.nodeFeatures

    x.copy(nodeFeatures = residual)
  }

  def state: Seq[(Constant, PTag)] =
    dropout.state ++ List(
      wNodeKey1,
      wNodeKey2,
      wEdgeKey,
      wNodeValue,
      wAttention
    ).map(
      _ -> GraphAttention.Weights
    )
}

object GraphAttention {

  case object Weights extends LeafTag

  case class Graph(
      nodeFeatures: Variable,
      edgeFeatures: Variable,
      edgeI: STen,
      edgeJ: STen
  )

  /** Graph Attention Network https://arxiv.org/pdf/1710.10903.pdf
    * Non-linearity in eq 4 and dropout is not applied to the final vertex activations
    *
    * Needs self edges to be already present in the graph
    *
    * @return next node representation (without relu, dropout) and a tensor with the original node and edge features ligned up like [N_i, N_j, E_ij]
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
      wAttention: Variable
  ) = {
    val nodeKey1 = nodeFeatures mm wNodeKey1
    val nodeKey2 = nodeFeatures mm wNodeKey2
    val edgeKey = edgeFeatures mm wEdgeKey
    val nodeValue = nodeFeatures mm wNodeValue

    val ninjeij = {
      Variable.cat(
        List(
          nodeKey1.indexSelect(dim = 0, const(edgeI)),
          nodeKey2.indexSelect(dim = 0, const(edgeJ)),
          edgeKey
        ),
        dim = 1
      )
    }

    val a = {
      val activations = (ninjeij mm wAttention).swish1
      val e = activations.exp
      val s = e.indexAdd(const(edgeJ), 0, nodeFeatures.shape(0))
      val sBroadCast = s.indexSelect(dim = 0, const(edgeJ))
      e / (sBroadCast + 1e-2)
    }.view(List(-1, wAttention.shape(1), 1))

    val numHeads = a.shape(1)
    assert(
      nodeValue.shape(1) % numHeads == 0,
      s"wNodeValue and wAttention size do not align ${wNodeValue
        .shape(1)} ${wAttention.shape(1)}"
    )

    val h = {
      val nodeValueScatter = nodeValue
        .indexSelect(dim = 0, const(edgeI))
        .view(List(-1, numHeads, nodeValue.shape(1) / numHeads))

      (a * nodeValueScatter)
        .view(List(-1, nodeValue.shape(1)))
        .indexAdd(const(edgeJ), 0, nodeFeatures.shape(0))

    }

    (h, ninjeij)
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
    m.wAttention.value.copyFrom(parameters(4))

  }

}
