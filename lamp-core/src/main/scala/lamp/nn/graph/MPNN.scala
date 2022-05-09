package lamp.nn.graph

import lamp.autograd._
import lamp._
import lamp.nn._

case class MPNN[M1 <: Module, M2 <: Module](
    messageTransform: M1 with Module,
    vertexTransform: M2 with Module,
    degreeNormalizeI: Boolean = true,
    degreeNormalizeJ: Boolean = true,
    aggregateJ: Boolean = true
) extends GraphModule {

  def state =
    messageTransform.state ++ vertexTransform.state

  override def forward[S: Sc](
      x: Graph
  ): Graph = {
    val message = {
      val vI = x.nodeFeatures.indexSelect(dim = 0, const(x.edgeI))
      val vJ = x.nodeFeatures.indexSelect(dim = 0, const(x.edgeJ))
      Variable.cat(List(x.edgeFeatures, vI, vJ), dim = 1)
    }
    val messageTx = messageTransform.forward(message)
    val aggregatedMessage = MPNN.aggregate(
      numVertices = x.nodeFeatures.shape(0),
      message = messageTx,
      edgeI = x.edgeI,
      edgeJ = x.edgeJ,
      degreeNormalizeI = degreeNormalizeI,
      degreeNormalizeJ = degreeNormalizeJ,
      aggregateJ = aggregateJ
    )
    val updatedVertex = vertexTransform.forward(
      Variable.cat(List(x.nodeFeatures, aggregatedMessage), dim = 1)
    )

    val addOrNot =
      if (updatedVertex.shape(1) == x.nodeFeatures.shape(1))
        x.nodeFeatures + updatedVertex
      else updatedVertex

    x.copy(nodeFeatures = addOrNot)

  }

}

object MPNN {

  implicit def trainingMode[
      M1 <: Module: TrainingMode,
      M2 <: Module: TrainingMode
  ]: TrainingMode[MPNN[M1, M2]] =
    TrainingMode
      .make[MPNN[M1, M2]](
        m =>
          m.copy(
            messageTransform = m.messageTransform.asEval,
            vertexTransform = m.vertexTransform.asEval
          ),
        m =>
          m.copy(
            messageTransform = m.messageTransform.asTraining,
            vertexTransform = m.vertexTransform.asTraining
          )
      )
  implicit def load[M1 <: Module: Load, M2 <: Module: Load]
      : Load[MPNN[M1, M2]] = Load.compose(_.messageTransform, _.vertexTransform)

  def countOccurences(t: STen, elems: Long)(implicit scope: Scope) = Scope {
    implicit scope =>
      val top = t.options
      val ones = STen.ones(t.shape, top)
      val zeros = STen.zeros(List(elems), top)
      zeros.indexAdd(0, t, ones)
  }

  def aggregate[S: Sc](
      numVertices: Long,
      message: Variable,
      edgeI: STen,
      edgeJ: STen,
      degreeNormalizeI: Boolean ,
      degreeNormalizeJ: Boolean ,
      aggregateJ: Boolean 
  ) = {
    val p = if (degreeNormalizeJ && degreeNormalizeI) -0.5 else -1d

    val normalizedMessage = {
      val t1 =
        if (degreeNormalizeI)
          message * const(
            countOccurences(edgeI, numVertices).pow(p).indexSelect(0, edgeI).view(-1L,1L)
          )
        else message

      if (degreeNormalizeJ)
        t1 * const(
          countOccurences(edgeJ, numVertices).pow(p).indexSelect(0, edgeJ).view(-1L,1L)
        )
      else t1
    }

    val aggregateI = normalizedMessage.indexAdd(const(edgeJ), 0, numVertices)

    if (aggregateJ) {
      val aggregateJ = normalizedMessage.indexAdd(const(edgeI), 0, numVertices)

      aggregateI + aggregateJ
    } else aggregateI

  }
}
