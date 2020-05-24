package candle
import org.saddle.Mat
package object autograd {
  def const(m: Mat[Double]): Variable = Constant(m).value.detached
  def param(m: Mat[Double]): Variable = Constant(m).value
}
