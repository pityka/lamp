package lamp.nn

import lamp.autograd.Variable
import lamp.Sc

case class Dropout(prob: Double, training: Boolean) extends Module {
  override def state = Nil
  def forward[S: Sc](x: Variable): Variable = x.dropout(prob, training)
}
object Dropout {
  implicit val load :Load[Dropout] = Load.identity[Dropout]
  implicit val tr : TrainingMode[Dropout] = TrainingMode
    .make[Dropout](_.copy(training = false), _.copy(training = true))
}
