package lamp.data

import lamp.autograd.{const}
import lamp._
import scala.collection.compat.immutable.ArraySeq

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
      rng: Option[scala.util.Random],
      transferBufferSize: Long
  ): BatchStream[(lamp.nn.graph.Graph,STen), Int, (BufferPair,BufferPair)] = {
    def makeNonEmptyBatch(
        idx: Array[Int],
        buffers: (BufferPair, BufferPair),
        device: Device
    ) = {
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
            c
          }
          t
        }
        val edgesV = {
          val t = Scope { implicit scope =>
            val c = STen.cat(
              ArraySeq.unsafeWrapArray(selectedGraphs.map(_.edgeFeatures)),
              0
            )
            c
          }
          t
        }
        val edgesIC =
          STen.cat(edgesI, 0)

        val edgesJC =
          STen.cat(edgesJ, 0)

        val graphIndicesV =
          STen.cat(graphIndices, 0)

        val selectedTargetOnDevice = Scope { implicit scope =>
          val idxT = STen.fromLongArray(
            idx.map(_.toLong),
            List(idx.length),
            targetPerGraph.device
          )
          val selectedTarget = targetPerGraph.index(idxT)
          device.to(selectedTarget)
        }

        val (floatBuffers, longBuffers) = buffers
        val onDeviceListFloats = device.toBatched(
          List(nodesV, edgesV),
          floatBuffers
        )
        val onDeviceListLongs = device.toBatched(
          List(edgesJC, edgesIC, graphIndicesV),
          longBuffers
        )

        StreamControl(
          (
            lamp.nn.graph.Graph(
              nodeFeatures = const(onDeviceListFloats(0)),
              edgeFeatures = const(onDeviceListFloats(1)),
              edgeJ = onDeviceListLongs(0),
              edgeI = onDeviceListLongs(1),
              vertexPoolingIndices = onDeviceListLongs(2)
            ),
            selectedTargetOnDevice
          )
        )
      }
    }

    val allocateBuffers = (device: Device) =>
      Scope.inResource.map({ implicit scope =>
        (
        device.allocateBuffers(transferBufferSize,STenOptions.f),
        device.allocateBuffers(transferBufferSize,STenOptions.l),
        )
      })

    val idx =
      rng
        .map(rng =>
          rng
            .shuffle(
              ArraySeq.unsafeWrapArray(
                Array.range(0, graphNodesAndEdges.length)
              )
            )
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

    BatchStream.fromIndicesWithBuffers(idx.toArray, allocateBuffers)(
      makeNonEmptyBatch
    )

  }

  /** Forms full batches of one big graph. Target is on node level
    *
    * Returns a pair of node features, edge list
    */
  def singleLargeGraph(
      graph: GraphBatchStream.Graph,
      targetPerNode: STen
  ): BatchStream[(lamp.nn.graph.Graph,STen), Boolean, Unit] =
    BatchStream.single(Scope.inResource.map { implicit scope =>
      StreamControl(
        (graph.toVariable, targetPerNode)
      )
    })

}
