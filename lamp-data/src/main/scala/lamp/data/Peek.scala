package lamp.data

import lamp.nn.Module
import lamp.autograd.Variable
import lamp.Sc
import lamp.nn.FW

case class Peek(label: String) extends Module {
  def state = Nil
  def forward[S: Sc,F:FW](x: Variable): Variable = {
    scribe.info(s"PEEK - $label - ${x.forward.shape} - ${x}")
    x
  }

}
