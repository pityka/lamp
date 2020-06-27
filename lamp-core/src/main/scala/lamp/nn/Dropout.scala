package lamp.nn

import lamp.autograd.Variable
import aten.Tensor

case class Dropout(prob: Double, training: Boolean) extends Module {
  override def state: Seq[(Variable, PTag)] = Nil
  def forward(x: Variable): Variable = x.dropout(prob, training)
}
object Dropout {
  implicit val load = Load.identity[Dropout]
  implicit val tr = TrainingMode
    .make[Dropout](_.copy(training = false), _.copy(training = true))
}
