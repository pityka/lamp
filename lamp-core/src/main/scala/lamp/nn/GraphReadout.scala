package lamp.nn

import lamp.autograd._
import aten.ATen

case class GraphReadout[M <: GraphModule](
    m: M with GraphModule
) extends GenericModule[
      (Variable, Variable, Variable),
      Variable
    ] {

  def state: Seq[(Variable, PTag)] =
    m.state

  def forward(
      x: (Variable, Variable, Variable)
  ): Variable = {
    import lamp.syntax
    val (nodes, edges, graphIndices) = x
    val (mN, mE) = m.forward((nodes, edges))
    val expanded =
      graphIndices.view(List(-1, 1)).expandAs(mN.value)
    val max = ATen.max_2(graphIndices.value)
    val maxi = max.toLongMat.raw(0) + 1
    max.release
    mN.scatterAdd(expanded, 0, maxi).releaseWithVariable(edges, mE)
  }

}

object GraphReadout {

  implicit def trainingMode[M1 <: GraphModule: TrainingMode] =
    TrainingMode.make[GraphReadout[M1]](
      module => GraphReadout(module.m.asEval),
      module => GraphReadout(module.m.asTraining)
    )

  implicit def load[M1 <: GraphModule: Load] =
    Load.make[GraphReadout[M1]](module =>
      tensors => {
        GraphReadout(module.m.load(tensors))
      }
    )

}
