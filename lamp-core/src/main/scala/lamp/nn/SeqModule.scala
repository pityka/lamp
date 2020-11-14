package lamp.nn

import lamp.Sc

object sequence {

  def apply[T1, T2, T3, M1 <: GenericModule[T1, T2], M2 <: GenericModule[
    T2,
    T3
  ]](
      m1: M1 with GenericModule[T1, T2],
      m2: M2 with GenericModule[T2, T3]
  ) = Seq2(m1, m2)

  def apply[T1, T2, T3, T4, M1 <: GenericModule[
    T1,
    T2
  ], M2 <: GenericModule[
    T2,
    T3
  ], M3 <: GenericModule[T3, T4]](
      m1: M1 with GenericModule[T1, T2],
      m2: M2 with GenericModule[T2, T3],
      m3: M3 with GenericModule[T3, T4]
  ) = Seq3(m1, m2, m3)

  def apply[
      T1,
      T2,
      T3,
      T4,
      T5,
      M1 <: GenericModule[
        T1,
        T2,
      ],
      M2 <: GenericModule[
        T2,
        T3,
      ],
      M3 <: GenericModule[T3, T4],
      M4 <: GenericModule[T4, T5]
  ](
      m1: M1 with GenericModule[T1, T2],
      m2: M2 with GenericModule[T2, T3],
      m3: M3 with GenericModule[T3, T4],
      m4: M4 with GenericModule[T4, T5]
  ) = Seq4(m1, m2, m3, m4)

  def apply[
      T1,
      T2,
      T3,
      T4,
      T5,
      T6,
      M1 <: GenericModule[
        T1,
        T2,
      ],
      M2 <: GenericModule[
        T2,
        T3,
      ],
      M3 <: GenericModule[T3, T4],
      M4 <: GenericModule[T4, T5],
      M5 <: GenericModule[T5, T6]
  ](
      m1: M1 with GenericModule[T1, T2],
      m2: M2 with GenericModule[T2, T3],
      m3: M3 with GenericModule[T3, T4],
      m4: M4 with GenericModule[T4, T5],
      m5: M5 with GenericModule[T5, T6]
  ) = Seq5(m1, m2, m3, m4, m5)

  def apply[
      T1,
      T2,
      T3,
      T4,
      T5,
      T6,
      T7,
      M1 <: GenericModule[
        T1,
        T2,
      ],
      M2 <: GenericModule[
        T2,
        T3,
      ],
      M3 <: GenericModule[T3, T4],
      M4 <: GenericModule[T4, T5],
      M5 <: GenericModule[T5, T6],
      M6 <: GenericModule[T6, T7]
  ](
      m1: M1 with GenericModule[T1, T2],
      m2: M2 with GenericModule[T2, T3],
      m3: M3 with GenericModule[T3, T4],
      m4: M4 with GenericModule[T4, T5],
      m5: M5 with GenericModule[T5, T6],
      m6: M6 with GenericModule[T6, T7]
  ) = Seq6(m1, m2, m3, m4, m5, m6)
}

case class Seq2[T1, T2, T3, M1 <: GenericModule[T1, T2], M2 <: GenericModule[
  T2,
  T3
]](
    m1: M1 with GenericModule[T1, T2],
    m2: M2 with GenericModule[T2, T3]
) extends GenericModule[T1, T3] {

  override def state =
    m1.state.map { case (param, ptag)   => (param, Sequential.Tag(ptag, 0)) } ++
      m2.state.map { case (param, ptag) => (param, Sequential.Tag(ptag, 1)) }

  def forward[S: Sc](x: T1) = m2.forward(m1.forward(x))

}

object Seq2 {
  implicit def trainingMode[T1, T2, T3, S1, S2, M1 <: GenericModule[
    T1,
    T2
  ]: TrainingMode, M2 <: GenericModule[
    T2,
    T3
  ]: TrainingMode] =
    TrainingMode.make[Seq2[T1, T2, T3, M1, M2]](
      module => Seq2(module.m1.asEval, module.m2.asEval),
      module => Seq2(module.m1.asTraining, module.m2.asTraining)
    )

  implicit def load[T1, T2, T3, S1, S2, M1 <: GenericModule[
    T1,
    T2
  ]: Load, M2 <: GenericModule[
    T2,
    T3
  ]: Load] =
    Load.make[Seq2[T1, T2, T3, M1, M2]](module =>
      tensors => {
        val m1S = module.m1.state.size
        val m2S = module.m2.state.size

        module.m1.load(tensors.take(m1S))
        module.m2.load(tensors.drop(m1S).take(m2S))
      }
    )

}

case class Seq3[T1, T2, T3, T4, M1 <: GenericModule[
  T1,
  T2
], M2 <: GenericModule[
  T2,
  T3
], M3 <: GenericModule[T3, T4]](
    m1: M1 with GenericModule[T1, T2],
    m2: M2 with GenericModule[T2, T3],
    m3: M3 with GenericModule[T3, T4]
) extends GenericModule[T1, T4] {

  override def state =
    m1.state.map { case (param, ptag)   => (param, Sequential.Tag(ptag, 0)) } ++
      m2.state.map { case (param, ptag) => (param, Sequential.Tag(ptag, 1)) } ++
      m3.state.map { case (param, ptag) => (param, Sequential.Tag(ptag, 2)) }

  def forward[S: Sc](x: T1) = {
    m3.forward(m2.forward(m1.forward(x)))
  }

}

object Seq3 {
  implicit def trainingMode[T1, T2, T3, T4, M1 <: GenericModule[
    T1,
    T2,
  ]: TrainingMode, M2 <: GenericModule[
    T2,
    T3,
  ]: TrainingMode, M3 <: GenericModule[
    T3,
    T4,
  ]: TrainingMode] =
    TrainingMode.make[Seq3[T1, T2, T3, T4, M1, M2, M3]](
      module => Seq3(module.m1.asEval, module.m2.asEval, module.m3.asEval),
      module =>
        Seq3(
          module.m1.asTraining,
          module.m2.asTraining,
          module.m3.asTraining
        )
    )

  implicit def load[T1, T2, T3, T4, M1 <: GenericModule[
    T1,
    T2,
  ]: Load, M2 <: GenericModule[
    T2,
    T3,
  ]: Load, M3 <: GenericModule[
    T3,
    T4,
  ]: Load] =
    Load.make[Seq3[T1, T2, T3, T4, M1, M2, M3]](module =>
      tensors => {
        val m1S = module.m1.state.size
        val m2S = module.m2.state.size
        val m3S = module.m3.state.size

        module.m1.load(tensors.take(m1S))
        module.m2.load(tensors.drop(m1S).take(m2S))
        module.m3.load(tensors.drop(m1S + m2S).take(m3S))
      }
    )

}

case class Seq4[
    T1,
    T2,
    T3,
    T4,
    T5,
    M1 <: GenericModule[
      T1,
      T2,
    ],
    M2 <: GenericModule[
      T2,
      T3,
    ],
    M3 <: GenericModule[T3, T4],
    M4 <: GenericModule[T4, T5]
](
    m1: M1 with GenericModule[T1, T2],
    m2: M2 with GenericModule[T2, T3],
    m3: M3 with GenericModule[T3, T4],
    m4: M4 with GenericModule[T4, T5]
) extends GenericModule[T1, T5] {

  override def state =
    m1.state.map { case (param, ptag)   => (param, Sequential.Tag(ptag, 0)) } ++
      m2.state.map { case (param, ptag) => (param, Sequential.Tag(ptag, 1)) } ++
      m3.state.map { case (param, ptag) => (param, Sequential.Tag(ptag, 2)) } ++
      m4.state.map { case (param, ptag) => (param, Sequential.Tag(ptag, 3)) }

  def forward[S: Sc](x: T1) = {
    m4.forward(m3.forward(m2.forward(m1.forward(x))))
  }

}

object Seq4 {
  implicit def trainingMode[
      T1,
      T2,
      T3,
      T4,
      T5,
      M1 <: GenericModule[
        T1,
        T2,
      ]: TrainingMode,
      M2 <: GenericModule[
        T2,
        T3,
      ]: TrainingMode,
      M3 <: GenericModule[
        T3,
        T4,
      ]: TrainingMode,
      M4 <: GenericModule[
        T4,
        T5,
      ]: TrainingMode
  ] =
    TrainingMode
      .make[Seq4[T1, T2, T3, T4, T5, M1, M2, M3, M4]](
        module =>
          Seq4(
            module.m1.asEval,
            module.m2.asEval,
            module.m3.asEval,
            module.m4.asEval
          ),
        module =>
          Seq4(
            module.m1.asTraining,
            module.m2.asTraining,
            module.m3.asTraining,
            module.m4.asTraining
          )
      )

  implicit def load[T1, T2, T3, T4, T5, M1 <: GenericModule[
    T1,
    T2,
  ]: Load, M2 <: GenericModule[
    T2,
    T3,
  ]: Load, M3 <: GenericModule[
    T3,
    T4,
  ]: Load, M4 <: GenericModule[
    T4,
    T5,
  ]: Load] =
    Load.make[Seq4[T1, T2, T3, T4, T5, M1, M2, M3, M4]](module =>
      tensors => {
        val m1S = module.m1.state.size
        val m2S = module.m2.state.size
        val m3S = module.m3.state.size
        val m4S = module.m4.state.size

        module.m1.load(tensors.take(m1S))
        module.m2.load(tensors.drop(m1S).take(m2S))
        module.m3.load(tensors.drop(m1S + m2S).take(m3S))
        module.m4.load(tensors.drop(m1S + m2S + m3S).take(m4S))
      }
    )

}

case class Seq5[
    T1,
    T2,
    T3,
    T4,
    T5,
    T6,
    M1 <: GenericModule[
      T1,
      T2,
    ],
    M2 <: GenericModule[
      T2,
      T3,
    ],
    M3 <: GenericModule[T3, T4],
    M4 <: GenericModule[T4, T5],
    M5 <: GenericModule[T5, T6]
](
    m1: M1 with GenericModule[T1, T2],
    m2: M2 with GenericModule[T2, T3],
    m3: M3 with GenericModule[T3, T4],
    m4: M4 with GenericModule[T4, T5],
    m5: M5 with GenericModule[T5, T6]
) extends GenericModule[T1, T6] {

  override def state =
    m1.state.map { case (param, ptag)   => (param, Sequential.Tag(ptag, 0)) } ++
      m2.state.map { case (param, ptag) => (param, Sequential.Tag(ptag, 1)) } ++
      m3.state.map { case (param, ptag) => (param, Sequential.Tag(ptag, 2)) } ++
      m4.state.map { case (param, ptag) => (param, Sequential.Tag(ptag, 3)) } ++
      m5.state.map { case (param, ptag) => (param, Sequential.Tag(ptag, 4)) }

  def forward[S: Sc](x: T1) = {
    m5.forward(m4.forward(m3.forward(m2.forward(m1.forward(x)))))
  }

}
object Seq5 {
  implicit def trainingMode[
      T1,
      T2,
      T3,
      T4,
      T5,
      T6,
      M1 <: GenericModule[
        T1,
        T2,
      ]: TrainingMode,
      M2 <: GenericModule[
        T2,
        T3,
      ]: TrainingMode,
      M3 <: GenericModule[
        T3,
        T4,
      ]: TrainingMode,
      M4 <: GenericModule[
        T4,
        T5,
      ]: TrainingMode,
      M5 <: GenericModule[
        T5,
        T6,
      ]: TrainingMode
  ] =
    TrainingMode
      .make[Seq5[
        T1,
        T2,
        T3,
        T4,
        T5,
        T6,
        M1,
        M2,
        M3,
        M4,
        M5
      ]](
        module =>
          Seq5(
            module.m1.asEval,
            module.m2.asEval,
            module.m3.asEval,
            module.m4.asEval,
            module.m5.asEval
          ),
        module =>
          Seq5(
            module.m1.asTraining,
            module.m2.asTraining,
            module.m3.asTraining,
            module.m4.asTraining,
            module.m5.asTraining
          )
      )

  implicit def load[
      T1,
      T2,
      T3,
      T4,
      T5,
      T6,
      M1 <: GenericModule[
        T1,
        T2,
      ]: Load,
      M2 <: GenericModule[
        T2,
        T3,
      ]: Load,
      M3 <: GenericModule[
        T3,
        T4,
      ]: Load,
      M4 <: GenericModule[
        T4,
        T5,
      ]: Load,
      M5 <: GenericModule[
        T5,
        T6,
      ]: Load
  ] =
    Load.make[Seq5[
      T1,
      T2,
      T3,
      T4,
      T5,
      T6,
      M1,
      M2,
      M3,
      M4,
      M5
    ]](module =>
      tensors => {
        val m1S = module.m1.state.size
        val m2S = module.m2.state.size
        val m3S = module.m3.state.size
        val m4S = module.m4.state.size
        val m5S = module.m5.state.size

        module.m1.load(tensors.take(m1S))
        module.m2.load(tensors.drop(m1S).take(m2S))
        module.m3.load(tensors.drop(m1S + m2S).take(m3S))
        module.m4.load(tensors.drop(m1S + m2S + m3S).take(m4S))
        module.m5.load(tensors.drop(m1S + m2S + m3S + m4S).take(m5S))
      }
    )

}

case class Seq6[
    T1,
    T2,
    T3,
    T4,
    T5,
    T6,
    T7,
    M1 <: GenericModule[
      T1,
      T2,
    ],
    M2 <: GenericModule[
      T2,
      T3,
    ],
    M3 <: GenericModule[T3, T4],
    M4 <: GenericModule[T4, T5],
    M5 <: GenericModule[T5, T6],
    M6 <: GenericModule[T6, T7]
](
    m1: M1 with GenericModule[T1, T2],
    m2: M2 with GenericModule[T2, T3],
    m3: M3 with GenericModule[T3, T4],
    m4: M4 with GenericModule[T4, T5],
    m5: M5 with GenericModule[T5, T6],
    m6: M6 with GenericModule[T6, T7]
) extends GenericModule[T1, T7] {

  override def state =
    m1.state.map { case (param, ptag)   => (param, Sequential.Tag(ptag, 0)) } ++
      m2.state.map { case (param, ptag) => (param, Sequential.Tag(ptag, 1)) } ++
      m3.state.map { case (param, ptag) => (param, Sequential.Tag(ptag, 2)) } ++
      m4.state.map { case (param, ptag) => (param, Sequential.Tag(ptag, 3)) } ++
      m5.state.map { case (param, ptag) => (param, Sequential.Tag(ptag, 4)) } ++
      m6.state.map { case (param, ptag) => (param, Sequential.Tag(ptag, 5)) }

  def forward[S: Sc](x: T1) = {
    m6.forward(m5.forward(m4.forward(m3.forward(m2.forward(m1.forward(x))))))
  }

}
object Seq6 {
  implicit def trainingMode[
      T1,
      T2,
      T3,
      T4,
      T5,
      T6,
      T7,
      M1 <: GenericModule[
        T1,
        T2,
      ]: TrainingMode,
      M2 <: GenericModule[
        T2,
        T3,
      ]: TrainingMode,
      M3 <: GenericModule[
        T3,
        T4,
      ]: TrainingMode,
      M4 <: GenericModule[
        T4,
        T5,
      ]: TrainingMode,
      M5 <: GenericModule[
        T5,
        T6,
      ]: TrainingMode,
      M6 <: GenericModule[
        T6,
        T7,
      ]: TrainingMode
  ] =
    TrainingMode
      .make[Seq6[
        T1,
        T2,
        T3,
        T4,
        T5,
        T6,
        T7,
        M1,
        M2,
        M3,
        M4,
        M5,
        M6
      ]](
        module =>
          Seq6(
            module.m1.asEval,
            module.m2.asEval,
            module.m3.asEval,
            module.m4.asEval,
            module.m5.asEval,
            module.m6.asEval
          ),
        module =>
          Seq6(
            module.m1.asTraining,
            module.m2.asTraining,
            module.m3.asTraining,
            module.m4.asTraining,
            module.m5.asTraining,
            module.m6.asTraining
          )
      )

  implicit def load[
      T1,
      T2,
      T3,
      T4,
      T5,
      T6,
      T7,
      M1 <: GenericModule[
        T1,
        T2,
      ]: Load,
      M2 <: GenericModule[
        T2,
        T3,
      ]: Load,
      M3 <: GenericModule[
        T3,
        T4,
      ]: Load,
      M4 <: GenericModule[
        T4,
        T5,
      ]: Load,
      M5 <: GenericModule[
        T5,
        T6,
      ]: Load,
      M6 <: GenericModule[
        T6,
        T7,
      ]: Load
  ] =
    Load.make[Seq6[
      T1,
      T2,
      T3,
      T4,
      T5,
      T6,
      T7,
      M1,
      M2,
      M3,
      M4,
      M5,
      M6
    ]](module =>
      tensors => {
        val m1S = module.m1.state.size
        val m2S = module.m2.state.size
        val m3S = module.m3.state.size
        val m4S = module.m4.state.size
        val m5S = module.m5.state.size
        val m6S = module.m6.state.size

        module.m1.load(tensors.take(m1S))
        module.m2.load(tensors.drop(m1S).take(m2S))
        module.m3.load(tensors.drop(m1S + m2S).take(m3S))
        module.m4.load(tensors.drop(m1S + m2S + m3S).take(m4S))
        module.m5.load(tensors.drop(m1S + m2S + m3S + m4S).take(m5S))
        module.m6.load(tensors.drop(m1S + m2S + m3S + m4S + m5S).take(m6S))
      }
    )

}
