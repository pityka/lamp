package lamp.data

import lamp.autograd.{const}
import lamp._
import scala.collection.immutable.ArraySeq

object GraphBatchStream {

  case class Graph(
      nodeFeatures: STen,
      edgeFeatures: STen,
      edgeI: STen,
      edgeJ: STen
  ) {
    def toVariable[S: Sc] = lamp.nn.graph.Graph(
      const(nodeFeatures),
      const(edgeFeatures),
      edgeI,
      edgeJ,
      STen.zeros(List(1))
    )
  }

  /** Forms minibatches from multiple small graphs. Assumes target is graph
    * level.
    *
    * Returns a triple of node features, edge list, graph indices
    */
  def smallGraphStream(
      minibatchSize: Int,
      graphNodesAndEdges: Array[Graph],
      targetPerGraph: STen,
      rng: Option[scala.util.Random]
  ): BatchStream[lamp.nn.graph.Graph, Int] = {
    def makeNonEmptyBatch(idx: Array[Int], device: Device) = {
      Scope.inResource.map { implicit scope =>
        val selectedGraphs = idx.map(graphNodesAndEdges)
        val (
          _,
          _,
          edgesI,
          edgesJ,
          graphIndices
        ) = selectedGraphs
          .foldLeft(
            (
              0L,
              0L,
              Vector.empty[STen],
              Vector.empty[STen],
              Vector.empty[STen]
            )
          ) {
            case (
                  (
                    offset,
                    graphIndex,
                    edgeAccumulatorI,
                    edgeAccumulatorJ,
                    graphIndicesAccumulator
                  ),
                  nextGraph
                ) =>
              val numNodes = nextGraph.nodeFeatures.shape(0)
              val mappedEdgeI = nextGraph.edgeI + offset
              val mappedEdgeJ = nextGraph.edgeJ + offset

              val graphIndices = {
                val z = STen.zeros(List(numNodes), STenOptions.l)
                z + graphIndex
              }

              (
                offset + numNodes,
                graphIndex + 1,
                edgeAccumulatorI :+ mappedEdgeI,
                edgeAccumulatorJ :+ mappedEdgeJ,
                graphIndicesAccumulator :+ graphIndices
              )
          }
        val nodesV = {
          val t = Scope { implicit scope =>
            val c = STen.cat(
              ArraySeq.unsafeWrapArray(selectedGraphs.map(_.nodeFeatures)),
              0
            )
            device.to(c)
          }
          const(t)
        }
        val edgesV = {
          val t = Scope { implicit scope =>
            val c = STen.cat(
              ArraySeq.unsafeWrapArray(selectedGraphs.map(_.edgeFeatures)),
              0
            )
            device.to(c)
          }
          const(t)
        }
        val edgesIC = Scope { implicit scope =>
          device.to(STen.cat(edgesI, 0))
        }
        val edgesJC = Scope { implicit scope =>
          device.to(STen.cat(edgesJ, 0))
        }

        val graphIndicesV =
          Scope { implicit scope =>
            device.to(STen.cat(graphIndices, 0))
          }

        val selectedTargetOnDevice = Scope { implicit scope =>
          val idxT = STen.fromLongArray(
            idx.map(_.toLong),
            List(idx.length),
            targetPerGraph.device
          )
          val selectedTarget = targetPerGraph.index(idxT)
          device.to(selectedTarget)
        }

        StreamControl(
          (
            lamp.nn.graph.Graph(
              nodeFeatures = nodesV,
              edgeFeatures = edgesV,
              edgeJ = edgesJC,
              edgeI = edgesIC,
              vertexPoolingIndices = graphIndicesV
            ),
            selectedTargetOnDevice
          )
        )
      }
    }

    val idx =
      rng
        .map(rng =>
          rng
            .shuffle(Array.range(0, graphNodesAndEdges.length))
            .grouped(minibatchSize)
            .map(_.toArray)
            .toList
        )
        .getOrElse(
          Array
            .range(0, graphNodesAndEdges.length)
            .grouped(minibatchSize)
            .toList
        )

    BatchStream.fromIndices(idx.toArray)(makeNonEmptyBatch)

  }

  /** Forms full batches of one big graph. Target is on node level
    *
    * Returns a pair of node features, edge list
    */
  def singleLargeGraph(
      graph: GraphBatchStream.Graph,
      targetPerNode: STen
  ): BatchStream[lamp.nn.graph.Graph, Boolean] =
    BatchStream.single(Scope.inResource.map { implicit scope =>
      StreamControl(
        (graph.toVariable, targetPerNode)
      )
    })

}
