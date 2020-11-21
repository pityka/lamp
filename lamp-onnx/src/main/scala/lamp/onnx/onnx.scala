package lamp.onnx

import lamp.autograd.Variable

case class VariableInfo(
    variable: Variable,
    name: String,
    input: Boolean,
    docString: String = ""
)
