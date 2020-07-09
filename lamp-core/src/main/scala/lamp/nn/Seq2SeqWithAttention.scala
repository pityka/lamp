package lamp.nn

import lamp.autograd.Variable
import lamp.autograd.ConcatenateAddNewDim
import lamp.util.NDArray

case class Seq2SeqWithAttention[S0, S1, M0 <: Module, M1 <: StatefulModule2[
  Variable,
  Variable,
  S0,
  S1
], M2 <: StatefulModule[
  Variable,
  Variable,
  S1
]](
    destinationEmbedding: M0 with Module,
    encoder: M1 with StatefulModule2[Variable, Variable, S0, S1],
    decoder: M2 with StatefulModule[Variable, Variable, S1],
    padToken: Long
)(val stateToKey: S1 => Variable)
    extends StatefulModule2[(Variable, Variable), Variable, S0, S1] {

  def attentionDecoder(keyValue: Variable, source: Variable) =
    AttentionDecoder(
      decoder,
      destinationEmbedding,
      stateToKey,
      keyValue,
      source,
      padToken
    )

  override def forward(x: ((Variable, Variable), S0)): (Variable, S1) = {
    val ((source, dest), state0) = x
    val embeddedDest = destinationEmbedding.forward(dest)
    val (encoderOutput, encoderState) = encoder.forward((source, state0))

    Attention.forward(
      decoder = decoder,
      x = embeddedDest,
      keyValue = encoderOutput,
      state = encoderState,
      tokens = source,
      padToken = padToken
    )(stateToKey)
  }

  override def state: Seq[(Variable, PTag)] =
    destinationEmbedding.state ++ encoder.state ++ decoder.state

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
      ]: TrainingMode
  ]: TrainingMode[Seq2SeqWithAttention[S0, S1, M0, M1, M2]] =
    TrainingMode.make[Seq2SeqWithAttention[S0, S1, M0, M1, M2]](
      m =>
        Seq2SeqWithAttention(
          m.destinationEmbedding.asEval,
          m.encoder.asEval,
          m.decoder.asEval,
          m.padToken
        )(m.stateToKey),
      m =>
        Seq2SeqWithAttention(
          m.destinationEmbedding.asTraining,
          m.encoder.asTraining,
          m.decoder.asTraining,
          m.padToken
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
      ]: Load
  ]: Load[Seq2SeqWithAttention[S0, S1, M0, M1, M2]] =
    Load.make[Seq2SeqWithAttention[S0, S1, M0, M1, M2]] { m => t =>
      val m0Size = m.destinationEmbedding.state.size
      val mESize = m.encoder.state.size
      val mDSize = m.decoder.state.size
      Seq2SeqWithAttention(
        m.destinationEmbedding.load(t.take(m0Size)),
        m.encoder.load(t.drop(m0Size).take(mESize)),
        m.decoder.load(t.drop(mESize + m0Size).take(mDSize)),
        m.padToken
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
      ]
  ](
      implicit is: InitState[M1, S0]
  ): InitState[Seq2SeqWithAttention[S0, S1, M0, M1, M2], S0] =
    InitState.make[Seq2SeqWithAttention[S0, S1, M0, M1, M2], S0] { m =>
      m.encoder.initState
    }
}
