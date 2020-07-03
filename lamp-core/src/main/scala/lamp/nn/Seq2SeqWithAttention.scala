package lamp.nn

import lamp.autograd.Variable

case class Seq2SeqWithAttention[S0, S1, M0 <: Module, M1 <: StatefulModule2[
  Variable,
  Variable,
  S0,
  S1
], M2 <: StatefulModule[
  Variable,
  Variable,
  S1
], M3 <: Module](
    destinationEmbedding: M0 with Module,
    encoder: M1 with StatefulModule2[Variable, Variable, S0, S1],
    decoder: M2 with StatefulModule[Variable, Variable, S1],
    contextLinear: M3 with Module
)(val stateToKey: S1 => Variable)
    extends StatefulModule2[(Variable, Variable), Variable, S0, S1] {

  override def forward(x: ((Variable, Variable), S0)): (Variable, S1) = {
    val ((source, dest), state0) = x
    val embeddedDest = destinationEmbedding.forward(dest)
    val (encoderOutput, encoderState) = encoder.forward((source, state0))
    Attention.forward(
      decoder = decoder,
      x = embeddedDest,
      keyValue = encoderOutput,
      contextModule = contextLinear,
      state = encoderState
    )(stateToKey)
  }

  override def state: Seq[(Variable, PTag)] =
    destinationEmbedding.state ++ encoder.state ++ decoder.state ++ contextLinear.state

}

object Seq2SeqWithAttention {
  implicit def trainingMode[
      S0,
      S1,
      M0 <: Module: TrainingMode,
      M1 <: StatefulModule2[
        Variable,
        Variable,
        S0,
        S1
      ]: TrainingMode,
      M2 <: StatefulModule[
        Variable,
        Variable,
        S1
      ]: TrainingMode,
      M3 <: Module: TrainingMode
  ]: TrainingMode[Seq2SeqWithAttention[S0, S1, M0, M1, M2, M3]] =
    TrainingMode.make[Seq2SeqWithAttention[S0, S1, M0, M1, M2, M3]](
      m =>
        Seq2SeqWithAttention(
          m.destinationEmbedding.asEval,
          m.encoder.asEval,
          m.decoder.asEval,
          m.contextLinear.asEval
        )(m.stateToKey),
      m =>
        Seq2SeqWithAttention(
          m.destinationEmbedding.asTraining,
          m.encoder.asTraining,
          m.decoder.asTraining,
          m.contextLinear.asTraining
        )(m.stateToKey)
    )
  implicit def load[
      S0,
      S1,
      M0 <: Module: TrainingMode: Load,
      M1 <: StatefulModule2[Variable, Variable, S0, S1]: Load,
      M2 <: StatefulModule[
        Variable,
        Variable,
        S1
      ]: Load,
      M3 <: Module: Load
  ]: Load[Seq2SeqWithAttention[S0, S1, M0, M1, M2, M3]] =
    Load.make[Seq2SeqWithAttention[S0, S1, M0, M1, M2, M3]] { m => t =>
      val m0Size = m.destinationEmbedding.state.size
      val mESize = m.encoder.state.size
      val mDSize = m.decoder.state.size
      val mCSize = m.contextLinear.state.size
      Seq2SeqWithAttention(
        m.destinationEmbedding.load(t.take(m0Size)),
        m.encoder.load(t.drop(m0Size).take(mESize)),
        m.decoder.load(t.drop(mESize + m0Size).take(mDSize)),
        m.contextLinear.load(t.drop(mESize + m0Size + mDSize).take(mCSize))
      )(m.stateToKey)
    }
  implicit def initState[
      S0,
      S1,
      M0 <: Module: TrainingMode,
      M1 <: StatefulModule2[
        Variable,
        Variable,
        S0,
        S1
      ],
      M2 <: StatefulModule[
        Variable,
        Variable,
        S1
      ],
      M3 <: Module
  ](
      implicit is: InitState[M1, S0]
  ): InitState[Seq2SeqWithAttention[S0, S1, M0, M1, M2, M3], S0] =
    InitState.make[Seq2SeqWithAttention[S0, S1, M0, M1, M2, M3], S0] { m =>
      m.encoder.initState
    }
}
