package lamp
import scala.language.implicitConversions
import lamp.autograd.Variable

package object nn {
  implicit def funToModule(fun: Variable => Variable) = FunctionModule(fun)
}
