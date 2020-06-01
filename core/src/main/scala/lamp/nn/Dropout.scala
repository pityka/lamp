package lamp.nn

import lamp.autograd.Variable

case class Dropout(prob: Double, training: Boolean) extends Module {
  def parameters: Seq[(Variable, PTag)] = Nil
  def forward(x: Variable): Variable = x.dropout(prob, training)
  override def asTraining = copy(training = true)
  override def asEval = copy(training = false)
}
