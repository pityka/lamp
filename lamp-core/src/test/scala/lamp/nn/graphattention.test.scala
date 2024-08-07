package lamp.nn

import org.scalatest.funsuite.AnyFunSuite
import lamp.autograd._

import lamp.util.NDArray
import lamp.Scope
import lamp.STen._
import lamp.nn.graph.GraphAttention
import lamp.nn.graph.Graph
import lamp.STen
import lamp.saddle._
import org.saddle.doubleOrd

class GraphAttentionSuite extends AnyFunSuite {

  test("graph attention") {
    Scope.root { implicit scope =>
      /** node features, 5 x 3
        */
      val nodes = NDArray
        .tensorFromLongNDArray(
          NDArray(
            Array(0L, 1L, 0L, 1L, 0L, 0L, 1L, 0L, 0L, 1L, 0L, 0L, 1L, 0L, 0L),
            List(5, 3)
          )
        )
        .owned
        .castToDouble

      val graphIndices = STen.zeros(List(5))

      /** edge features, 6 x 2
        */
      val edges = NDArray
        .tensorFromLongNDArray(
          NDArray(
            Array.fill(22)(1L),
            List(11, 2)
          )
        )
        .owned
        .castToDouble

      /** edge indices, 11 x 2
        */
      val edgeIdx = NDArray
        .tensorFromLongNDArray(
          NDArray(
            Array(0L, 1L, 1L, 0L, 3L, 4L, 3L, 2L, 3L, 4L, 4L, 4L, 0L, 0L, 1L,
              1L, 2L, 2L, 3L, 3L, 4L, 4L),
            List(11, 2)
          )
        )
        .owned

      /** wNodeKey1, 3 x 4
        */
      val wNodeKey1 = NDArray
        .tensorFromLongNDArray(
          NDArray(
            Array(0L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L),
            List(3, 4)
          )
        )
        .owned
        .castToDouble

      /** wNodeKey2, 3 x 4
        */
      val wNodeKey2 = NDArray
        .tensorFromLongNDArray(
          NDArray(
            Array(1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L),
            List(3, 4)
          )
        )
        .owned
        .castToDouble

      /** 3 x 6 (3 * 2head)
        */
      val wNodeValue = NDArray
        .tensorFromLongNDArray(
          NDArray(
            Array(1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 2L, 2L, 2L, 2L, 2L, 2L,
              2L, 2L, 2L),
            List(3, 6)
          )
        )
        .owned
        .castToDouble
      // 2 x 4
      val wEdgeKey = NDArray
        .tensorFromLongNDArray(
          NDArray(
            Array(1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L),
            List(2, 4)
          )
        )
        .owned
        .castToDouble
      val wEdgeKey2 = NDArray
        .tensorFromLongNDArray(
          NDArray(
            Array(1L, 1L, 1L, 1L),
            List(2, 2)
          )
        )
        .owned
        .castToDouble
      // 12 x 4
      val wAttention = NDArray
        .tensorFromLongNDArray(
          NDArray(
            Array.fill(12)(1L),
            List(6, 2)
          )
        )
        .owned
        .castToDouble

      val result = GraphAttention
        .multiheadGraphAttention(
          nodeFeatures = const(nodes),
          edgeFeatures = const(edges),
          edgeI = edgeIdx.select(1, 0),
          edgeJ = edgeIdx.select(1, 1),
          wNodeKey1 = const(wNodeKey1),
          wNodeKey2 = const(wNodeKey2),
          wEdgeKey = const(wEdgeKey),
          wNodeValue = const(wNodeValue),
          wAttention = Some(const(wAttention)),
          numHeads = 2
        ).value
        .toMat
      assert(result.numRows == 5)
      assert(result.numCols == 6)
      assert(result.roundTo(10).raw(0, 0) == 1.0)
      assert(result.roundTo(10).raw(0, 3) == 1.5)
      val resultDot = GraphAttention
        .multiheadGraphAttention(
          nodeFeatures = const(nodes),
          edgeFeatures = const(edges),
          edgeI = edgeIdx.select(1, 0),
          edgeJ = edgeIdx.select(1, 1),
          wNodeKey1 = const(wNodeKey1),
          wNodeKey2 = const(wNodeKey2),
          wEdgeKey = const(wEdgeKey2),
          wNodeValue = const(wNodeValue),
          wAttention = None,
          numHeads = 2
        ).value
        .toMat
      assert(resultDot.numRows == 5)
      assert(resultDot.numCols == 6)
      assert(resultDot.roundTo(10).raw(0, 0) == 1.0)
      assert(resultDot.roundTo(10).raw(0, 3) == 1.5)

      assert(
        GraphAttention
          .apply(
            nodeDim = 3,
            edgeDim = 2,
            attentionKeyHiddenDimPerHead = 2,
            attentionNumHeads = 2,
            valueDimPerHead = 3,
            dropout = 0d,
            tOpt = nodes.options,
            dotProductAttention = false,
            nonLinearity = false
          )
          .forward(
            Graph(
              const(nodes),
              const(edges),
              edgeIdx.select(1, 0),
              edgeIdx.select(1, 1),
              graphIndices
            )
          )
          .nodeFeatures
          .shape == List(5, 6)
      )

      assert(
        GraphAttention
          .apply(
            nodeDim = 3,
            edgeDim = 2,
            attentionKeyHiddenDimPerHead = 2,
            attentionNumHeads = 2,
            valueDimPerHead = 3,
            dropout = 0d,
            tOpt = nodes.options,
            dotProductAttention = true,
            nonLinearity = false
          )
          .forward(
            Graph(
              const(nodes),
              const(edges),
              edgeIdx.select(1, 0),
              edgeIdx.select(1, 1),
              graphIndices
            )
          )
          .nodeFeatures
          .shape == List(5, 6)
      )

      ()
    }
  }

}
