package lamp.nn

import lamp.autograd.Variable
import lamp.Sc

case class ResidualModule[M <: Module](
    transform: M with Module
) extends Module {

  def state =
    transform.state

  override def forward[S: Sc](
      x: Variable
  ): Variable = {
    val n = transform.forward(x)
    if (n.sizes == x.sizes) n + x
    else n
  }

}

object ResidualModule {

  implicit def trainingMode[M <: Module: TrainingMode] : TrainingMode[ResidualModule[M]] =
    TrainingMode
      .make[ResidualModule[M]](
        m => m.copy(transform = m.transform.asEval),
        m => m.copy(transform = m.transform.asTraining)
      )
  implicit def load[M <: Module: Load] : Load[ResidualModule[M]] = Load.make[ResidualModule[M]] {
    m => tensors => m.transform.load(tensors)

  }

}
