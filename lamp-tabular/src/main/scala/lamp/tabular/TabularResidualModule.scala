package lamp.tabular

import lamp.nn._
import lamp.autograd.Variable
import lamp.Sc
import lamp.Scope
import lamp.STenOptions
import lamp.autograd.GraphConfiguration

case class TabularResidual[Block <: Module, B2 <: Module](
    straight: B2 with Module,
    block: Block with Module
) extends Module {
  override def state = straight.state ++ block.state
  def forward[S: Sc](x: Variable) = {
    val r = straight.forward(x)
    val l = block.forward(x)
    (r + l)
  }

}

object TabularResidual {
  def make(
      inChannels: Int,
      hiddenChannels: Int,
      outChannels: Int,
      tOpt: STenOptions,
      dropout: Double
  )(implicit scope: Scope, conf: GraphConfiguration) =
    TabularResidual(
      straight = sequence(
        BatchNorm(inChannels, tOpt),
        Linear(inChannels, outChannels, tOpt)
      ),
      block = sequence(
        sequence(
          BatchNorm(inChannels, tOpt),
          Dropout(dropout * 0.2, training = true),
          Linear(inChannels, math.max(outChannels, hiddenChannels), tOpt),
          Fun(implicit scope => _.relu)
        ),
        sequence(
          BatchNorm(math.max(outChannels, hiddenChannels), tOpt),
          Dropout(dropout, training = true),
          Linear(
            math.max(outChannels, hiddenChannels),
            math.max(outChannels, hiddenChannels / 2),
            tOpt
          ),
          Fun(implicit scope => _.relu)
        ),
        sequence(
          BatchNorm(math.max(outChannels, hiddenChannels / 2), tOpt),
          Dropout(dropout * 0.2, training = true),
          Linear(math.max(outChannels, hiddenChannels / 2), outChannels, tOpt)
        )
      )
    )

  implicit def trainingMode[
      M2 <: Module: TrainingMode,
      B2 <: Module: TrainingMode
  ] =
    TrainingMode.make[TabularResidual[M2, B2]](
      m => TabularResidual(m.straight.asEval, m.block.asEval),
      m => TabularResidual(m.straight.asTraining, m.block.asTraining)
    )
  implicit def load[M2 <: Module: Load, B2 <: Module: Load] =
    Load.make[TabularResidual[M2, B2]] { m => t =>
      m.straight.load(t.take(m.straight.state.size))
      m.block.load(t.drop(m.straight.state.size).take(m.block.state.size))
    }

}
