package lamp.data

import cats.effect._
import aten.Tensor
import org.saddle._
import lamp.autograd.{TensorHelpers, const}
import aten.ATen
import lamp.Device
import lamp.autograd.Variable
import lamp.autograd.AllocatedVariablePool

object GraphBatchStream {

  /** Forms minibatches from multiple small graphs. Assumes target is graph level.
    *
    * Returns a triple of node features, edge list, graph indices
    */
  def smallGraphMode(
      minibatchSize: Int,
      graphNodesAndEdges: Vec[(Tensor, Tensor)],
      targetPerGraph: Tensor,
      device: Device,
      rng: org.saddle.spire.random.Generator
  )(
      implicit pool: AllocatedVariablePool
  ): BatchStream[(Variable, Variable, Variable)] = {
    def makeNonEmptyBatch(idx: Array[Int]) = {
      Resource.make(IO {
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
              Seq.empty[Tensor],
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
              val edges = TensorHelpers
                .toLongMat(nextEdges)
                .rows
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
          val c = ATen.cat(nodes.toArray, 0)
          val cd = device.to(c)
          c.release
          const(cd).releasable
        }
        val edgesV = {
          val i = TensorHelpers.fromLongVec(edgesI.toVec, device)
          val j = TensorHelpers.fromLongVec(edgesJ.toVec, device)
          val r = const(ATen.stack(Array(i, j), 1)).releasable
          i.release
          j.release
          r
        }
        val graphIndicesV = const(
          TensorHelpers.fromLongVec(graphIndices.toVec, device)
        ).releasable

        val idxT = TensorHelpers.fromLongVec(idx.toVec.map(_.toLong))
        val selectedTarget = ATen.index(targetPerGraph, Array(idxT))
        val selectedTargetOnDevice = device.to(selectedTarget)
        selectedTarget.release
        idxT.release

        Option(
          ((nodesV, edgesV, graphIndicesV), selectedTargetOnDevice)
        )
      }) {
        case None => IO.unit
        case Some((_, b)) =>
          IO {
            b.release
          }
      }
    }
    val emptyResource = Resource
      .pure[IO, Option[((Variable, Variable, Variable), Tensor)]](None)

    val idx =
      array
        .shuffle(array.range(0, graphNodesAndEdges.length), rng)
        .grouped(minibatchSize)
        .toList
    new BatchStream[(Variable, Variable, Variable)] {
      private var remaining = idx
      def nextBatch: Resource[IO, Option[
        ((Variable, Variable, Variable), Tensor)
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
      nodes: Tensor,
      edges: Tensor,
      targetPerNode: Tensor
  )(
      implicit pool: AllocatedVariablePool
  ): BatchStream[(Variable, Variable)] =
    new BatchStream[(Variable, Variable)] {
      var i = 1
      def nextBatch: Resource[IO, Option[
        ((Variable, Variable), Tensor)
      ]] = {
        if (i == 1) {
          i -= 1
          Resource.make(IO {
            val nodesV = const(nodes)
            val edgesV = const(edges)

            Option(
              ((nodesV, edgesV), targetPerNode)
            )
          }) {
            case _ => IO.unit

          }
        } else { Resource.make(IO(None))(_ => IO.unit) }
      }
    }

}
