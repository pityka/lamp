package lamp.nn

import lamp.autograd._
import aten.ATen
import lamp.Sc

case class GraphReadout[M <: GraphModule](
    m: M with GraphModule,
    pooling: GraphReadout.PoolType
) extends GenericModule[
      (Variable, Variable, Variable),
      Variable
    ] {

  def state: Seq[(Variable, PTag)] =
    m.state

  def forward[S: Sc](
      x: (Variable, Variable, Variable)
  ): Variable = {
    import lamp.util.syntax
    val (nodes, edges, graphIndices) = x
    val (mN, _) = m.forward((nodes, edges))

    val max = ATen.max_2(graphIndices.value)

    val maxi = max.toLongMat.raw(0) + 1
    max.release
    val sum = mN.indexAdd(graphIndices, 0, maxi)
    pooling match {
      case GraphReadout.Sum => sum
      case GraphReadout.Mean =>
        val ones =
          const(ATen.ones(Array(nodes.shape(0), 1), sum.options))(sum.pool)
        val counts = ones.indexAdd(graphIndices, 0, maxi)
        sum / counts
    }
  }

}

object GraphReadout {

  sealed trait PoolType
  case object Sum extends PoolType
  case object Mean extends PoolType

  implicit def trainingMode[M1 <: GraphModule: TrainingMode] =
    TrainingMode.make[GraphReadout[M1]](
      module => GraphReadout(module.m.asEval, module.pooling),
      module => GraphReadout(module.m.asTraining, module.pooling)
    )

  implicit def load[M1 <: GraphModule: Load] =
    Load.make[GraphReadout[M1]](module =>
      tensors => {
        GraphReadout(module.m.load(tensors), module.pooling)
      }
    )

}
