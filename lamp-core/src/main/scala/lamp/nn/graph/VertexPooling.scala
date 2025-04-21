package lamp.nn.graph

import lamp.autograd._
import lamp.nn.FW
import aten.ATen
import lamp.Sc
import lamp.STen

object VertexPooling {

  def apply[S:Sc,F:FW](
      x: Graph,
      pooling: PoolType
  ): Variable = {

    val max = ATen.max_1(x.vertexPoolingIndices.value)

    val maxi = lamp.TensorHelpers.toLongArray(max).apply(0) + 1
    max.release
    val sum = x.nodeFeatures.indexAdd((x.vertexPoolingIndices), 0, maxi)
    pooling match {
      case VertexPooling.Sum => sum
      case VertexPooling.Mean =>
        val ones =
          const(STen.ones(List(x.nodeFeatures.shape.apply(0), 1), sum.options))
        val counts = ones.indexAdd((x.vertexPoolingIndices), 0, maxi)
        sum / counts
    }
  }
  sealed trait PoolType
  case object Sum extends PoolType
  case object Mean extends PoolType

}
