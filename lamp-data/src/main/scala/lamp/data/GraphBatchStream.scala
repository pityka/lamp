package lamp.data

import cats.effect._
import org.saddle._
import lamp.autograd.{const}
import lamp.Device
import lamp.autograd.Variable
import lamp.Scope
import lamp.STen

object GraphBatchStream {

  /** Forms minibatches from multiple small graphs. Assumes target is graph level.
    *
    * Returns a triple of node features, edge list, graph indices
    */
  def smallGraphMode(
      minibatchSize: Int,
      graphNodesAndEdges: Vec[(STen, STen)],
      targetPerGraph: STen,
      device: Device,
      rng: Option[org.saddle.spire.random.Generator]
  ): BatchStream[(Variable, Variable, Variable)] = {
    def makeNonEmptyBatch(idx: Array[Int]) = {
      Scope.inResource.map { implicit scope =>
        val selectedGraphs = graphNodesAndEdges.take(idx).toSeq
        val (
          _,
          _,
          nodes,
          edgesI,
          edgesJ,
          graphIndices
        ) = selectedGraphs
          .foldLeft(
            (
              0L,
              0L,
              Seq.empty[STen],
              Vector.empty[Long],
              Vector.empty[Long],
              Vector.empty[Long]
            )
          ) {
            case (
                (
                  offset,
                  graphIndex,
                  nodeAccumalator,
                  edgeAccumulatorI,
                  edgeAccumulatorJ,
                  graphIndicesAccumulator
                ),
                (nextNodes, nextEdges)
                ) =>
              val numNodes = nextNodes.sizes.head.toInt
              val edges = nextEdges.toLongMat.rows
                .map(v => v.raw(0) -> v.raw(1))
              assert(edges.map(_._1).forall(e => e >= 0 && e < numNodes))
              assert(edges.map(_._2).forall(e => e >= 0 && e < numNodes))
              val mappedEdges = edges.map {
                case (i, j) => (i + offset, j + offset)
              }
              val graphIndices =
                (0 until numNodes map (_ => graphIndex)).toVector

              val newOffset = offset + numNodes
              val newGraphIndex = graphIndex + 1
              val newNodeAcc = nodeAccumalator :+ nextNodes
              val newEdgeAccI = edgeAccumulatorI ++ mappedEdges.map(_._1)
              val newEdgeAccJ = edgeAccumulatorJ ++ mappedEdges.map(_._2)
              val newGraphIndices = graphIndicesAccumulator ++ graphIndices
              (
                newOffset,
                newGraphIndex,
                newNodeAcc,
                newEdgeAccI,
                newEdgeAccJ,
                newGraphIndices
              )
          }
        val nodesV = {
          val t = Scope { implicit scope =>
            val c = STen.cat(nodes, 0)
            device.to(c)
          }
          const(t)
        }
        val edgesV = {
          val s = Scope { implicit scope =>
            val i = STen.fromLongVec(edgesI.toVec, device)
            val j = STen.fromLongVec(edgesJ.toVec, device)
            STen.stack(Array(i, j), 1)
          }
          const(s)
        }
        val graphIndicesV = const(
          STen.fromLongVec(graphIndices.toVec, device)
        )

        val selectedTargetOnDevice = Scope { implicit scope =>
          val idxT = STen.fromLongVec(idx.toVec.map(_.toLong))
          val selectedTarget = targetPerGraph.index(idxT)
          device.to(selectedTarget)
        }

        Option(
          ((nodesV, edgesV, graphIndicesV), selectedTargetOnDevice)
        )
      }
    }
    val emptyResource = Resource
      .pure[IO, Option[((Variable, Variable, Variable), STen)]](None)

    val idx =
      rng
        .map(rng =>
          array
            .shuffle(array.range(0, graphNodesAndEdges.length), rng)
            .grouped(minibatchSize)
            .toList
        )
        .getOrElse(
          array
            .range(0, graphNodesAndEdges.length)
            .grouped(minibatchSize)
            .toList
        )
    new BatchStream[(Variable, Variable, Variable)] {
      private var remaining = idx
      def nextBatch: Resource[IO, Option[
        ((Variable, Variable, Variable), STen)
      ]] =
        remaining match {
          case Nil => emptyResource
          case x :: tail =>
            val r = makeNonEmptyBatch(x)
            remaining = tail
            r
        }
    }

  }

  /** Forms full batches of one big graph. Target is on node level
    *
    * Returns a pair of node features, edge list
    */
  def bigGraphModeFullBatch(
      nodes: STen,
      edges: STen,
      targetPerNode: STen
  ): BatchStream[(Variable, Variable)] =
    new BatchStream[(Variable, Variable)] {
      var i = 1
      def nextBatch: Resource[IO, Option[
        ((Variable, Variable), STen)
      ]] = {
        if (i == 1) {
          i -= 1
          Scope.inResource.map { _ =>
            val nodesV = const(nodes)
            val edgesV = const(edges)

            Option(
              ((nodesV, edgesV), targetPerNode)
            )
          }
        } else { Resource.make(IO(None))(_ => IO.unit) }
      }
    }

}
