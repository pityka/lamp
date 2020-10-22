package lamp.data

import lamp.nn.Module
import lamp.autograd.Variable
import lamp.Sc

case class Peek(label: String) extends Module {
  def state = Nil
  def forward[S: Sc](x: Variable): Variable = {
    scribe.info(s"PEEK - $label - ${x.shape} - ${x}")
    x
  }

}
