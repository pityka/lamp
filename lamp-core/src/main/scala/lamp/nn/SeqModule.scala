package lamp.nn

import lamp.autograd.Variable
import aten.Tensor

case class Seq2[T1, T2](
    m1: StatefulModule[T1],
    m2: StatefulModule[T2]
) extends StatefulModule[(T1, T2)] {

  type Me = Seq2[T1, T2]

  override def asEval: Me = Seq2(m1.asEval, m2.asEval)
  override def asTraining: Me = Seq2(m1.asTraining, m2.asTraining)
  override def state =
    m1.state.map { case (param, ptag)   => (param, Sequential.Tag(ptag, 0)) } ++
      m2.state.map { case (param, ptag) => (param, Sequential.Tag(ptag, 1)) }

  def forward1(x: Variable, st: (T1, T2)) = {
    val (x1, t1) = m1.forward1(x, st._1)
    val (x2, t2) = m2.forward1(x1, st._2)
    (x2, (t1, t2))
  }

  override def load(tensors: Seq[Tensor]) = {
    val m1S = m1.state.size
    val m2S = m2.state.size

    val loaded1 = m1.load(tensors.take(m1S))
    val loaded2 = m2.load(tensors.drop(m1S).take(m2S))

    Seq2(loaded1, loaded2)
  }

}

case class Seq3[T1, T2, T3](
    m1: StatefulModule[T1],
    m2: StatefulModule[T2],
    m3: StatefulModule[T3]
) extends StatefulModule[(T1, T2, T3)] {

  type Me = Seq3[T1, T2, T3]

  override def asEval: Me = Seq3(m1.asEval, m2.asEval, m3.asEval)
  override def asTraining: Me = Seq3(m1.asTraining, m2.asTraining, m3.asEval)
  override def state =
    m1.state.map { case (param, ptag)   => (param, Sequential.Tag(ptag, 0)) } ++
      m2.state.map { case (param, ptag) => (param, Sequential.Tag(ptag, 1)) } ++
      m3.state.map { case (param, ptag) => (param, Sequential.Tag(ptag, 2)) }

  def forward1(x: Variable, st: (T1, T2, T3)) = {
    val (x1, t1) = m1.forward1(x, st._1)
    val (x2, t2) = m2.forward1(x1, st._2)
    val (x3, t3) = m3.forward1(x2, st._3)
    (x3, (t1, t2, t3))
  }

  override def load(tensors: Seq[Tensor]) = {
    val m1S = m1.state.size
    val m2S = m2.state.size
    val m3S = m2.state.size

    val loaded1 = m1.load(tensors.take(m1S))
    val loaded2 = m2.load(tensors.drop(m1S).take(m2S))
    val loaded3 = m3.load(tensors.drop(m1S + m2S).take(m3S))

    Seq3(loaded1, loaded2, loaded3)
  }

}
case class Seq4[T1, T2, T3, T4](
    m1: StatefulModule[T1],
    m2: StatefulModule[T2],
    m3: StatefulModule[T3],
    m4: StatefulModule[T4]
) extends StatefulModule[(T1, T2, T3, T4)] {

  type Me = Seq4[T1, T2, T3, T4]

  override def asEval: Me = Seq4(m1.asEval, m2.asEval, m3.asEval, m4.asEval)
  override def asTraining: Me =
    Seq4(m1.asTraining, m2.asTraining, m3.asTraining, m4.asTraining)
  override def state =
    m1.state.map { case (param, ptag)   => (param, Sequential.Tag(ptag, 0)) } ++
      m2.state.map { case (param, ptag) => (param, Sequential.Tag(ptag, 1)) } ++
      m3.state.map { case (param, ptag) => (param, Sequential.Tag(ptag, 2)) } ++
      m4.state.map { case (param, ptag) => (param, Sequential.Tag(ptag, 3)) }

  def forward1(x: Variable, st: (T1, T2, T3, T4)) = {
    val (x1, t1) = m1.forward1(x, st._1)
    val (x2, t2) = m2.forward1(x1, st._2)
    val (x3, t3) = m3.forward1(x2, st._3)
    val (x4, t4) = m4.forward1(x3, st._4)
    (x4, (t1, t2, t3, t4))
  }

  override def load(tensors: Seq[Tensor]) = {
    val m1S = m1.state.size
    val m2S = m2.state.size
    val m3S = m3.state.size
    val m4S = m4.state.size

    val loaded1 = m1.load(tensors.take(m1S))
    val loaded2 = m2.load(tensors.drop(m1S).take(m2S))
    val loaded3 = m3.load(tensors.drop(m1S + m2S).take(m3S))
    val loaded4 = m4.load(tensors.drop(m1S + m2S + m3S).take(m4S))

    Seq4(loaded1, loaded2, loaded3, loaded4)
  }

}
