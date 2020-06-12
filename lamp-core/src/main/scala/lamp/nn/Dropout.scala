package lamp.nn

import lamp.autograd.Variable
import aten.Tensor

case class Dropout(prob: Double, training: Boolean) extends Module {
  override def load(parameters: Seq[Tensor]) = this
  override def state: Seq[(Variable, PTag)] = Nil
  def forward(x: Variable): Variable = x.dropout(prob, training)
  override def asTraining = copy(training = true)
  override def asEval = copy(training = false)
}
