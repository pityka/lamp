package lamp.data

import lamp.nn.Module
import lamp.autograd.Variable

case class Peek(label: String) extends Module {

  def forward(x: Variable): Variable = {
    scribe.info(s"PEEK - $label - ${x.shape} - ${x}")
    x
  }

}
