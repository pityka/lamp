package lamp.autograd
import aten.{Tensor, ATen}
import java.{util => ju}
// import scala.collection.mutable
import lamp.FloatingPointPrecision
import lamp.util.syntax
import lamp.Scope
import lamp.Sc

/**
  * Params: the input and the function which calculates the partial derivative
  * of the function value wrt to this input
  *
  * y = f1 o f2 o .. o fn
  *
  * One of these subexpression (f_i) has value w2 and arguments w1.
  * We can write this: dy/dw1 = dy/dw2 * dw2/dw1.
  * dw2/dw1 is the Jacobian of f_i at the current value of w1.
  * dy/dw2 is the Jacobian of y wrt to w2 at the current value of w2.
  *
  * The current value of w1 and w2 are computed in a forward pass.
  * The value dy/dy is 1 and from this dy/dw2 is recursed in the backward pass.
  * The Jacobian function of dw2/dw1 is either computed symbolically.
  *
  * https://en.wikipedia.org/wiki/Automatic_differentiation#Reverse_accumulation
  * http://www.cs.cmu.edu/~wcohen/10-605/notes/autodiff.pdf
  *
  * The function given in this argument is dy/dw2 => dy/dw2 * dw2/dw1.
  * The argument is coming down from the backward pass.
  * The Op fills in the symbolic part and the multiplication.
  *
  * The shape of the argument given to that function is the shape of the value of Op (dy/dw2)
  * The shape of the return is the shape of the argument (parameter) with respect the
  * derivative is taken (dy/dw1)
  *
  */
trait Op {
  val value: Variable
  val params: List[(Variable, (Tensor, Tensor) => Unit)]
}

object Variable {
  def apply(op: Op, value: Tensor, needsGrad: Boolean = true)(
      implicit sc: Scope
  ): Variable = Variable(op, value, sc, needsGrad)
}

case class Variable(
    op: Op,
    value: Tensor,
    scope: Scope,
    needsGrad: Boolean
) {
  scope(value)

  def pool = scope

  override def toString =
    s"Var(shape=$shape,value=$value,needsGrad=$needsGrad)"

  val options = value.options

  var partialDerivative: Option[Tensor] = None

  val sizes = value.sizes.toList

  def shape = sizes

  val id = ju.UUID.randomUUID()

  def needsNoGrad = copy(needsGrad = false)
  def detached = const(value)(scope)
  def zeroGrad() = {
    partialDerivative.foreach { t => ATen.zero_(t) }
  }

  lazy val wengert = Autograd.topologicalSort(this)

  def backprop(): Unit = {
    if (partialDerivative.isEmpty) {
      partialDerivative = Some(
        scope(ATen.ones_like(value, value.options))
      )
    }
    wengert.foreach { v =>
      v.op.params.foreach {
        case (v1, computeGrad) =>
          v1.accumulateGrad(v.partialDerivative.get, computeGrad)

      }
    }

  }

  def zipBackward(fn: (Tensor, Tensor) => Unit) = (this, fn)

  def accumulateGrad(
      incoming: Tensor,
      computeGrad: (Tensor, Tensor) => Unit
  ) = if (needsGrad) {

    if (partialDerivative.isEmpty) {
      partialDerivative = Some(scope(ATen.zeros(shape.toArray, options)))
    }
    computeGrad(incoming, partialDerivative.get)
  }

  def t[S: Sc] = Transpose(this).value
  def transpose[S: Sc](dim1: Int, dim2: Int) = Transpose(this, dim1, dim2).value
  def select[S: Sc](dim: Long, index: Long) =
    Select(this, dim = dim, index = index).value
  def indexSelect[S: Sc](dim: Long, index: Variable) =
    IndexSelect(this, dim = dim, index = index).value
  def argmax[S: Sc](dim: Long, keepDim: Boolean) =
    ArgMax(this, dim = dim, keepDim = keepDim).value
  def oneHot[S: Sc](numClasses: Int) =
    OneHot(this, numClasses).value
  def assign[S: Sc](other: Variable) =
    Assign(abandon = this, keep = other).value
  def maskFill[S: Sc](mask: Variable, fill: Double) =
    MaskFill(this, mask, fill).value
  def makeBooleanMask[S: Sc](q: Long) = EqWhere(this, q).value
  def cast[S: Sc](precision: FloatingPointPrecision) =
    CastToPrecision(this, precision).value
  def cat[S: Sc](other: Variable, dim: Long) =
    Concatenate(List(this, other), dim).value
  def +[S: Sc](other: Variable) = Add(this, other).value
  def +[S: Sc](other: Double) = ConstAdd(this, other).value
  def -[S: Sc](other: Variable) = Minus(this, other).value
  def *[S: Sc](other: Variable) = Mult(this, other).value
  def *[S: Sc](other: Double) = ConstMult(this, other).value
  def /[S: Sc](other: Variable) = Div(this, other).value
  def mm[S: Sc](other: Variable) = MatMul(this, other).value
  def bmm[S: Sc](other: Variable) = BatchedMatMul(this, other).value
  def relu[S: Sc] = Relu(this).value
  def gelu[S: Sc] = Gelu(this).value
  def sigmoid[S: Sc] = Sigmoid(this).value
  def dropout[S: Sc](prob: Double, train: Boolean) =
    Dropout(this, prob, train).value
  def scatterAdd[S: Sc](index: Variable, dim: Int, maxIndex: Long) =
    ScatterAdd(this, index, dim, maxIndex).value
  def indexAdd[S: Sc](index: Variable, dim: Int, maxIndex: Long) =
    IndexAdd(this, index, dim, maxIndex).value
  def sum[S: Sc] = Sum(this).value
  def expandAs[S: Sc](other: Tensor) = ExpandAs(this, other).value
  def rowSum[S: Sc] = RowSum(this).value
  def colSum[S: Sc] = ColSum(this).value
  def exp[S: Sc] = Exp(this).value
  def log[S: Sc] = Log(this).value
  def log1p[S: Sc] = Log1p(this).value
  def sin[S: Sc] = Sin(this).value
  def cos[S: Sc] = Cos(this).value
  def tan[S: Sc] = Tan(this).value
  def tanh[S: Sc] = Tanh(this).value
  def atan[S: Sc] = ArcTan(this).value
  def pow[S: Sc](const: Double) = PowConst(this, const).value
  def pow[S: Sc](exponent: Variable) = Pow(this, exponent).value
  def euclideanDistance[S: Sc](b: Variable, dim: Int) =
    EuclideanDistance(this, b, dim).value
  def logSoftMax[S: Sc](dim: Int) = LogSoftMax(this, dim).value
  def crossEntropy[S: Sc](other: Variable) =
    ((this.*(other)).rowSum).*(-1d)
  def nllLoss[S: Sc](
      target: Tensor,
      numClasses: Int,
      weights: Tensor,
      reduction: Reduction = Mean,
      ignore: Long = -100L
  ) =
    NllLoss(this, target, weights, numClasses, reduction, ignore).value
  def mseLoss[S: Sc](
      target: Tensor,
      reduction: Reduction = Mean
  ) =
    MseLoss(this, target, reduction).value
  def l1Loss[S: Sc](
      target: Tensor,
      reduction: Reduction = Mean
  ) =
    L1Loss(this, target, reduction).value
  def squaredFrobenius[S: Sc] = SquaredFrobeniusMatrixNorm(this).value
  def mean[S: Sc](dim: List[Int]) = Mean(this, dim).value
  def variance[S: Sc](dim: List[Int]) = Variance(this, dim).value
  def normalize[S: Sc](dim: List[Int]) = {
    (this - this.mean(dim)) / ((this.variance(dim) + 1e-6).pow(0.5))
  }
  def view[S: Sc](shape: List[Int]) =
    View(this, shape.map(_.toLong).toArray).value
  def flattenLastDimensions[S: Sc](dims: Int) =
    FlattenLastDimensions(this, dims).value

  def toMat = value.toMat
  def toLongMat = value.toLongMat
}

object Autograd {

  private[autograd] def topologicalSort[D](root: Variable): Seq[Variable] = {
    type V = Variable
    var order = List.empty[V]
    var marks = Set.empty[ju.UUID]
    var currentParents = Set.empty[ju.UUID]

    def visit(n: V): Unit =
      if (marks.contains(n.id)) ()
      else {
        if (currentParents.contains(n.id)) {
          println(s"error: loop to ${n.id}")
          ()
        } else {
          currentParents = currentParents + n.id
          val children = n.op.params.map(_._1)
          children.foreach(visit)
          currentParents = currentParents - n.id
          marks = marks + n.id
          order = n :: order
        }
      }

    visit(root)

    order

  }

}
