package lamp.nn

import lamp.autograd.Variable
import aten.Tensor

object statefulSequence {
  def apply[T1, T2, T3, S1, S2, M1 <: StatefulModule[T1, T2, S1], M2 <: StatefulModule[
    T2,
    T3,
    S2
  ]](
      m1: M1 with StatefulModule[T1, T2, S1],
      m2: M2 with StatefulModule[T2, T3, S2]
  ) = StatefulSeq2(m1, m2)

  def apply[T1, T2, T3, T4, S1, S2, S3, M1 <: StatefulModule[
    T1,
    T2,
    S1
  ], M2 <: StatefulModule[
    T2,
    T3,
    S2
  ], M3 <: StatefulModule[T3, T4, S3]](
      m1: M1 with StatefulModule[T1, T2, S1],
      m2: M2 with StatefulModule[T2, T3, S2],
      m3: M3 with StatefulModule[T3, T4, S3]
  ) = StatefulSeq3(m1, m2, m3)

  def apply[
      T1,
      T2,
      T3,
      T4,
      T5,
      S1,
      S2,
      S3,
      S4,
      M1 <: StatefulModule[
        T1,
        T2,
        S1
      ],
      M2 <: StatefulModule[
        T2,
        T3,
        S2
      ],
      M3 <: StatefulModule[T3, T4, S3],
      M4 <: StatefulModule[T4, T5, S4]
  ](
      m1: M1 with StatefulModule[T1, T2, S1],
      m2: M2 with StatefulModule[T2, T3, S2],
      m3: M3 with StatefulModule[T3, T4, S3],
      m4: M4 with StatefulModule[T4, T5, S4]
  ) = StatefulSeq4(m1, m2, m3, m4)

  def apply[
      T1,
      T2,
      T3,
      T4,
      T5,
      T6,
      S1,
      S2,
      S3,
      S4,
      S5,
      M1 <: StatefulModule[
        T1,
        T2,
        S1
      ],
      M2 <: StatefulModule[
        T2,
        T3,
        S2
      ],
      M3 <: StatefulModule[T3, T4, S3],
      M4 <: StatefulModule[T4, T5, S4],
      M5 <: StatefulModule[T5, T6, S5]
  ](
      m1: M1 with StatefulModule[T1, T2, S1],
      m2: M2 with StatefulModule[T2, T3, S2],
      m3: M3 with StatefulModule[T3, T4, S3],
      m4: M4 with StatefulModule[T4, T5, S4],
      m5: M5 with StatefulModule[T5, T6, S5]
  ) = StatefulSeq5(m1, m2, m3, m4, m5)
}

case class StatefulSeq2[T1, T2, T3, S1, S2, M1 <: StatefulModule[T1, T2, S1], M2 <: StatefulModule[
  T2,
  T3,
  S2
]](
    m1: M1 with StatefulModule[T1, T2, S1],
    m2: M2 with StatefulModule[T2, T3, S2]
) extends StatefulModule[T1, T3, (S1, S2)] {

  override def state =
    m1.state.map { case (param, ptag)   => (param, Sequential.Tag(ptag, 0)) } ++
      m2.state.map { case (param, ptag) => (param, Sequential.Tag(ptag, 1)) }

  def forward(x: (T1, (S1, S2))) = forward1(x._1, x._2)
  def forward1(x: T1, st: (S1, S2)) = {
    val (x1, t1) = m1.forward((x, st._1))
    val (x2, t2) = m2.forward((x1, st._2))
    (x2, (t1, t2))
  }

}

object StatefulSeq2 {
  implicit def trainingMode[T1, T2, T3, S1, S2, M1 <: StatefulModule[
    T1,
    T2,
    S1
  ]: TrainingMode, M2 <: StatefulModule[
    T2,
    T3,
    S2
  ]: TrainingMode] =
    TrainingMode.make[StatefulSeq2[T1, T2, T3, S1, S2, M1, M2]](
      module => StatefulSeq2(module.m1.asEval, module.m2.asEval),
      module => StatefulSeq2(module.m1.asTraining, module.m2.asTraining)
    )

  implicit def load[T1, T2, T3, S1, S2, M1 <: StatefulModule[
    T1,
    T2,
    S1
  ]: Load, M2 <: StatefulModule[
    T2,
    T3,
    S2
  ]: Load] =
    Load.make[StatefulSeq2[T1, T2, T3, S1, S2, M1, M2]](module =>
      tensors => {
        val m1S = module.m1.state.size
        val m2S = module.m2.state.size

        val loaded1 = module.m1.load(tensors.take(m1S))
        val loaded2 = module.m2.load(tensors.drop(m1S).take(m2S))
        StatefulSeq2(loaded1, loaded2)
      }
    )

  implicit def initState[T1, T2, T3, S1, S2, M1 <: StatefulModule[
    T1,
    T2,
    S1
  ], M2 <: StatefulModule[
    T2,
    T3,
    S2
  ]](implicit is1: InitState[M1, S1], is2: InitState[M2, S2]) =
    InitState.make[StatefulSeq2[T1, T2, T3, S1, S2, M1, M2], (S1, S2)](module =>
      (module.m1.initState, module.m2.initState)
    )

}

case class StatefulSeq3[T1, T2, T3, T4, S1, S2, S3, M1 <: StatefulModule[
  T1,
  T2,
  S1
], M2 <: StatefulModule[
  T2,
  T3,
  S2
], M3 <: StatefulModule[T3, T4, S3]](
    m1: M1 with StatefulModule[T1, T2, S1],
    m2: M2 with StatefulModule[T2, T3, S2],
    m3: M3 with StatefulModule[T3, T4, S3]
) extends StatefulModule[T1, T4, (S1, S2, S3)] {

  override def state =
    m1.state.map { case (param, ptag)   => (param, Sequential.Tag(ptag, 0)) } ++
      m2.state.map { case (param, ptag) => (param, Sequential.Tag(ptag, 1)) } ++
      m3.state.map { case (param, ptag) => (param, Sequential.Tag(ptag, 2)) }

  def forward(x: (T1, (S1, S2, S3))) = forward1(x._1, x._2)
  def forward1(x: T1, st: (S1, S2, S3)) = {
    val (x1, t1) = m1.forward((x, st._1))
    val (x2, t2) = m2.forward((x1, st._2))
    val (x3, t3) = m3.forward((x2, st._3))
    (x3, (t1, t2, t3))
  }

}

object StatefulSeq3 {
  implicit def trainingMode[T1, T2, T3, T4, S1, S2, S3, M1 <: StatefulModule[
    T1,
    T2,
    S1
  ]: TrainingMode, M2 <: StatefulModule[
    T2,
    T3,
    S2
  ]: TrainingMode, M3 <: StatefulModule[
    T3,
    T4,
    S3
  ]: TrainingMode] =
    TrainingMode.make[StatefulSeq3[T1, T2, T3, T4, S1, S2, S3, M1, M2, M3]](
      module =>
        StatefulSeq3(module.m1.asEval, module.m2.asEval, module.m3.asEval),
      module =>
        StatefulSeq3(
          module.m1.asTraining,
          module.m2.asTraining,
          module.m3.asTraining
        )
    )

  implicit def load[T1, T2, T3, T4, S1, S2, S3, M1 <: StatefulModule[
    T1,
    T2,
    S1
  ]: Load, M2 <: StatefulModule[
    T2,
    T3,
    S2
  ]: Load, M3 <: StatefulModule[
    T3,
    T4,
    S3
  ]: Load] =
    Load.make[StatefulSeq3[T1, T2, T3, T4, S1, S2, S3, M1, M2, M3]](module =>
      tensors => {
        val m1S = module.m1.state.size
        val m2S = module.m2.state.size
        val m3S = module.m3.state.size

        val loaded1 = module.m1.load(tensors.take(m1S))
        val loaded2 = module.m2.load(tensors.drop(m1S).take(m2S))
        val loaded3 = module.m3.load(tensors.drop(m1S + m2S).take(m3S))
        StatefulSeq3(loaded1, loaded2, loaded3)
      }
    )

  implicit def initState[T1, T2, T3, T4, S1, S2, S3, M1 <: StatefulModule[
    T1,
    T2,
    S1
  ], M2 <: StatefulModule[
    T2,
    T3,
    S2
  ], M3 <: StatefulModule[
    T3,
    T4,
    S3
  ]](
      implicit is1: InitState[M1, S1],
      is2: InitState[M2, S2],
      is3: InitState[M3, S3]
  ) =
    InitState
      .make[StatefulSeq3[T1, T2, T3, T4, S1, S2, S3, M1, M2, M3], (S1, S2, S3)](
        module =>
          (module.m1.initState, module.m2.initState, module.m3.initState)
      )

}

case class StatefulSeq4[
    T1,
    T2,
    T3,
    T4,
    T5,
    S1,
    S2,
    S3,
    S4,
    M1 <: StatefulModule[
      T1,
      T2,
      S1
    ],
    M2 <: StatefulModule[
      T2,
      T3,
      S2
    ],
    M3 <: StatefulModule[T3, T4, S3],
    M4 <: StatefulModule[T4, T5, S4]
](
    m1: M1 with StatefulModule[T1, T2, S1],
    m2: M2 with StatefulModule[T2, T3, S2],
    m3: M3 with StatefulModule[T3, T4, S3],
    m4: M4 with StatefulModule[T4, T5, S4]
) extends StatefulModule[T1, T5, (S1, S2, S3, S4)] {

  override def state =
    m1.state.map { case (param, ptag)   => (param, Sequential.Tag(ptag, 0)) } ++
      m2.state.map { case (param, ptag) => (param, Sequential.Tag(ptag, 1)) } ++
      m3.state.map { case (param, ptag) => (param, Sequential.Tag(ptag, 2)) } ++
      m4.state.map { case (param, ptag) => (param, Sequential.Tag(ptag, 3)) }

  def forward(x: (T1, (S1, S2, S3, S4))) = forward1(x._1, x._2)
  def forward1(x: T1, st: (S1, S2, S3, S4)) = {
    val (x1, t1) = m1.forward((x, st._1))
    val (x2, t2) = m2.forward((x1, st._2))
    val (x3, t3) = m3.forward((x2, st._3))
    val (x4, t4) = m4.forward((x3, st._4))
    (x4, (t1, t2, t3, t4))
  }

}

object StatefulSeq4 {
  implicit def trainingMode[
      T1,
      T2,
      T3,
      T4,
      T5,
      S1,
      S2,
      S3,
      S4,
      M1 <: StatefulModule[
        T1,
        T2,
        S1
      ]: TrainingMode,
      M2 <: StatefulModule[
        T2,
        T3,
        S2
      ]: TrainingMode,
      M3 <: StatefulModule[
        T3,
        T4,
        S3
      ]: TrainingMode,
      M4 <: StatefulModule[
        T4,
        T5,
        S4
      ]: TrainingMode
  ] =
    TrainingMode
      .make[StatefulSeq4[T1, T2, T3, T4, T5, S1, S2, S3, S4, M1, M2, M3, M4]](
        module =>
          StatefulSeq4(
            module.m1.asEval,
            module.m2.asEval,
            module.m3.asEval,
            module.m4.asEval
          ),
        module =>
          StatefulSeq4(
            module.m1.asTraining,
            module.m2.asTraining,
            module.m3.asTraining,
            module.m4.asTraining
          )
      )

  implicit def load[T1, T2, T3, T4, T5, S1, S2, S3, S4, M1 <: StatefulModule[
    T1,
    T2,
    S1
  ]: Load, M2 <: StatefulModule[
    T2,
    T3,
    S2
  ]: Load, M3 <: StatefulModule[
    T3,
    T4,
    S3
  ]: Load, M4 <: StatefulModule[
    T4,
    T5,
    S4
  ]: Load] =
    Load.make[StatefulSeq4[T1, T2, T3, T4, T5, S1, S2, S3, S4, M1, M2, M3, M4]](
      module =>
        tensors => {
          val m1S = module.m1.state.size
          val m2S = module.m2.state.size
          val m3S = module.m3.state.size
          val m4S = module.m4.state.size

          val loaded1 = module.m1.load(tensors.take(m1S))
          val loaded2 = module.m2.load(tensors.drop(m1S).take(m2S))
          val loaded3 = module.m3.load(tensors.drop(m1S + m2S).take(m3S))
          val loaded4 = module.m4.load(tensors.drop(m1S + m2S + m3S).take(m4S))
          StatefulSeq4(loaded1, loaded2, loaded3, loaded4)
        }
    )

  implicit def initState[
      T1,
      T2,
      T3,
      T4,
      T5,
      S1,
      S2,
      S3,
      S4,
      M1 <: StatefulModule[
        T1,
        T2,
        S1
      ],
      M2 <: StatefulModule[
        T2,
        T3,
        S2
      ],
      M3 <: StatefulModule[
        T3,
        T4,
        S3
      ],
      M4 <: StatefulModule[
        T4,
        T5,
        S4
      ]
  ](
      implicit is1: InitState[M1, S1],
      is2: InitState[M2, S2],
      is3: InitState[M3, S3],
      is4: InitState[M4, S4]
  ) =
    InitState
      .make[StatefulSeq4[T1, T2, T3, T4, T5, S1, S2, S3, S4, M1, M2, M3, M4], (S1, S2, S3, S4)](
        module =>
          (
            module.m1.initState,
            module.m2.initState,
            module.m3.initState,
            module.m4.initState
          )
      )

}

case class StatefulSeq5[
    T1,
    T2,
    T3,
    T4,
    T5,
    T6,
    S1,
    S2,
    S3,
    S4,
    S5,
    M1 <: StatefulModule[
      T1,
      T2,
      S1
    ],
    M2 <: StatefulModule[
      T2,
      T3,
      S2
    ],
    M3 <: StatefulModule[T3, T4, S3],
    M4 <: StatefulModule[T4, T5, S4],
    M5 <: StatefulModule[T5, T6, S5]
](
    m1: M1 with StatefulModule[T1, T2, S1],
    m2: M2 with StatefulModule[T2, T3, S2],
    m3: M3 with StatefulModule[T3, T4, S3],
    m4: M4 with StatefulModule[T4, T5, S4],
    m5: M5 with StatefulModule[T5, T6, S5]
) extends StatefulModule[T1, T6, (S1, S2, S3, S4, S5)] {

  override def state =
    m1.state.map { case (param, ptag)   => (param, Sequential.Tag(ptag, 0)) } ++
      m2.state.map { case (param, ptag) => (param, Sequential.Tag(ptag, 1)) } ++
      m3.state.map { case (param, ptag) => (param, Sequential.Tag(ptag, 2)) } ++
      m4.state.map { case (param, ptag) => (param, Sequential.Tag(ptag, 3)) } ++
      m5.state.map { case (param, ptag) => (param, Sequential.Tag(ptag, 4)) }

  def forward(x: (T1, (S1, S2, S3, S4, S5))) = forward1(x._1, x._2)
  def forward1(x: T1, st: (S1, S2, S3, S4, S5)) = {
    val (x1, t1) = m1.forward((x, st._1))
    val (x2, t2) = m2.forward((x1, st._2))
    val (x3, t3) = m3.forward((x2, st._3))
    val (x4, t4) = m4.forward((x3, st._4))
    val (x5, t5) = m5.forward((x4, st._5))
    (x5, (t1, t2, t3, t4, t5))
  }

}
object StatefulSeq5 {
  implicit def trainingMode[
      T1,
      T2,
      T3,
      T4,
      T5,
      T6,
      S1,
      S2,
      S3,
      S4,
      S5,
      M1 <: StatefulModule[
        T1,
        T2,
        S1
      ]: TrainingMode,
      M2 <: StatefulModule[
        T2,
        T3,
        S2
      ]: TrainingMode,
      M3 <: StatefulModule[
        T3,
        T4,
        S3
      ]: TrainingMode,
      M4 <: StatefulModule[
        T4,
        T5,
        S4
      ]: TrainingMode,
      M5 <: StatefulModule[
        T5,
        T6,
        S5
      ]: TrainingMode
  ] =
    TrainingMode
      .make[StatefulSeq5[
        T1,
        T2,
        T3,
        T4,
        T5,
        T6,
        S1,
        S2,
        S3,
        S4,
        S5,
        M1,
        M2,
        M3,
        M4,
        M5
      ]](
        module =>
          StatefulSeq5(
            module.m1.asEval,
            module.m2.asEval,
            module.m3.asEval,
            module.m4.asEval,
            module.m5.asEval
          ),
        module =>
          StatefulSeq5(
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
      S1,
      S2,
      S3,
      S4,
      S5,
      M1 <: StatefulModule[
        T1,
        T2,
        S1
      ]: Load,
      M2 <: StatefulModule[
        T2,
        T3,
        S2
      ]: Load,
      M3 <: StatefulModule[
        T3,
        T4,
        S3
      ]: Load,
      M4 <: StatefulModule[
        T4,
        T5,
        S4
      ]: Load,
      M5 <: StatefulModule[
        T5,
        T6,
        S5
      ]: Load
  ] =
    Load.make[StatefulSeq5[
      T1,
      T2,
      T3,
      T4,
      T5,
      T6,
      S1,
      S2,
      S3,
      S4,
      S5,
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

        val loaded1 = module.m1.load(tensors.take(m1S))
        val loaded2 = module.m2.load(tensors.drop(m1S).take(m2S))
        val loaded3 = module.m3.load(tensors.drop(m1S + m2S).take(m3S))
        val loaded4 = module.m4.load(tensors.drop(m1S + m2S + m3S).take(m4S))
        val loaded5 =
          module.m5.load(tensors.drop(m1S + m2S + m3S + m4S).take(m5S))
        StatefulSeq5(loaded1, loaded2, loaded3, loaded4, loaded5)
      }
    )

  implicit def initState[
      T1,
      T2,
      T3,
      T4,
      T5,
      T6,
      S1,
      S2,
      S3,
      S4,
      S5,
      M1 <: StatefulModule[
        T1,
        T2,
        S1
      ],
      M2 <: StatefulModule[
        T2,
        T3,
        S2
      ],
      M3 <: StatefulModule[
        T3,
        T4,
        S3
      ],
      M4 <: StatefulModule[
        T4,
        T5,
        S4
      ],
      M5 <: StatefulModule[
        T5,
        T6,
        S5
      ]
  ](
      implicit is1: InitState[M1, S1],
      is2: InitState[M2, S2],
      is3: InitState[M3, S3],
      is4: InitState[M4, S4],
      is5: InitState[M5, S5]
  ) =
    InitState
      .make[StatefulSeq5[
        T1,
        T2,
        T3,
        T4,
        T5,
        T6,
        S1,
        S2,
        S3,
        S4,
        S5,
        M1,
        M2,
        M3,
        M4,
        M5
      ], (S1, S2, S3, S4, S5)](module =>
        (
          module.m1.initState,
          module.m2.initState,
          module.m3.initState,
          module.m4.initState,
          module.m5.initState
        )
      )

}
