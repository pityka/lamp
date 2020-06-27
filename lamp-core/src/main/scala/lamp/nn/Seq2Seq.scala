package lamp.nn

import lamp.autograd.Variable

case class Seq2Seq[S0, S1, M1 <: StatefulModule2[Variable, Variable, S0, S1], M2 <: StatefulModule[
  Variable,
  Variable,
  S1
]](
    encoder: M1 with StatefulModule2[Variable, Variable, S0, S1],
    decoder: M2 with StatefulModule[Variable, Variable, S1]
) extends StatefulModule2[(Variable, Variable), Variable, S0, S1] {

  override def forward(x: ((Variable, Variable), S0)): (Variable, S1) = {
    val ((source, dest), state0) = x
    val (encoderOutput, encoderState) = encoder.forward((source, state0))
    decoder.forward((dest.attach(encoderOutput), encoderState))
  }

  override def state: Seq[(Variable, PTag)] = encoder.state ++ decoder.state

}

object Seq2Seq {
  implicit def trainingMode[S0, S1, M1 <: StatefulModule2[
    Variable,
    Variable,
    S0,
    S1
  ]: TrainingMode, M2 <: StatefulModule[
    Variable,
    Variable,
    S1
  ]: TrainingMode]: TrainingMode[Seq2Seq[S0, S1, M1, M2]] =
    TrainingMode.make[Seq2Seq[S0, S1, M1, M2]](
      m => m.copy(m.encoder.asEval, m.decoder.asEval),
      m => m.copy(m.encoder.asTraining, m.decoder.asTraining)
    )
  implicit def load[S0, S1, M1 <: StatefulModule2[Variable, Variable, S0, S1]: Load, M2 <: StatefulModule[
    Variable,
    Variable,
    S1
  ]: Load]: Load[Seq2Seq[S0, S1, M1, M2]] =
    Load.make[Seq2Seq[S0, S1, M1, M2]] { m => t =>
      val mESize = m.encoder.state.size
      val mDSize = m.decoder.state.size
      m.copy(
        m.encoder.load(t.take(mESize)),
        m.decoder.load(t.drop(mESize).take(mDSize))
      )
    }
  implicit def initState[S0, S1, M1 <: StatefulModule2[
    Variable,
    Variable,
    S0,
    S1
  ], M2 <: StatefulModule[
    Variable,
    Variable,
    S1
  ]](implicit is: InitState[M1, S0]): InitState[Seq2Seq[S0, S1, M1, M2], S0] =
    InitState.make[Seq2Seq[S0, S1, M1, M2], S0] { m => m.encoder.initState }
}

case class WithInit[A, B, C, M <: StatefulModule[
  A,
  B,
  C
]](
    module: M with StatefulModule[A, B, C],
    init: C
) extends StatefulModule[A, B, C] {

  override def forward(x: (A, C)): (B, C) = {
    module.forward(x)
  }

  override def state: Seq[(Variable, PTag)] = module.state

}

object WithInit {
  implicit def trainingMode[A, B, C, M <: StatefulModule[
    A,
    B,
    C
  ]: TrainingMode] =
    TrainingMode.make[WithInit[A, B, C, M]](
      m => m.copy(m.module.asEval),
      m => m.copy(m.module.asTraining)
    )
  implicit def load[A, B, C, M <: StatefulModule[
    A,
    B,
    C
  ]: Load] =
    Load.make[WithInit[A, B, C, M]] { m => t =>
      m.copy(
        m.module.load(t)
      )
    }
  implicit def initState[A, B, C, M <: StatefulModule[
    A,
    B,
    C
  ]] =
    InitState.make[WithInit[A, B, C, M], C] { m => m.init }
}
