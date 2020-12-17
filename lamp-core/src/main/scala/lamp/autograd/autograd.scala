package lamp.autograd
import java.{util => ju}
import lamp.FloatingPointPrecision
import lamp.Scope
import lamp.Sc
import lamp.STen
import lamp.Movable

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
  val params: List[(Variable, (STen, STen) => Unit)]
}

object Variable {
  def apply(op: Op, value: STen)(
      implicit scope: Scope
  ): Variable =
    VariableNonConstant(
      op,
      value,
      STen.zerosLike(value)(scope)
    )

  def concatenateAddNewDim(inputs: Seq[Variable])(implicit scope: Scope) =
    new Stack(scope, inputs, 0).value

  def stack(inputs: Seq[Variable], dim: Int)(implicit scope: Scope) =
    new Stack(scope, inputs, dim).value

  def cat(inputs: Seq[Variable], dim: Long)(implicit scope: Scope) =
    new Concatenate(scope, inputs, dim).value
}

case class ConstantWithoutGrad(
    value: STen
) extends Constant {
  val partialDerivative = None
}

case class ConstantWithGrad(
    value: STen,
    pd: STen
) extends Constant {
  val partialDerivative = Some(pd)
}

sealed trait Constant extends Variable {
  final def op = None
}

object Constant {
  implicit val movable =
    Movable.nonEmpty[Constant] {
      case p: ConstantWithGrad    => List(p.value.value, p.pd.value)
      case p: ConstantWithoutGrad => List(p.value.value)
    }

}

case class VariableNonConstant(
    op1: Op,
    value: STen,
    pd: STen
) extends Variable {
  val op = Some(op1)
  val partialDerivative: Option[STen] = Some(pd)
}

object VariableNonConstant {
  implicit val movable =
    Movable.nonEmpty[VariableNonConstant](p =>
      p.wengert.flatMap {
        case ConstantWithGrad(value, pd)       => List(value.value, pd.value)
        case ConstantWithoutGrad(value)        => List(value.value)
        case VariableNonConstant(_, value, pd) => List(value.value, pd.value)
      } toList
    )
}

trait Variable {
  def op: Option[Op]
  def value: STen
  def partialDerivative: Option[STen]
  def needsGrad: Boolean = partialDerivative.isDefined

  override def toString =
    s"Var(shape=$shape,value=$value,needsGrad=$needsGrad)"

  def options[S: Sc] = value.options

  val sizes = value.sizes.toList

  def shape = sizes

  val id = ju.UUID.randomUUID()

  def detached = const(value)
  def zeroGrad() = {
    partialDerivative.foreach { t => t.zero_() }
  }

  lazy val wengert = Autograd.topologicalSort(this)

  def backprop(): Unit = {
    if (partialDerivative.isDefined) {
      partialDerivative.get.fill_(1d)
      wengert.foreach { v =>
        v.op.foreach(_.params.foreach {
          case (v1, computeGrad) =>
            v1.accumulateGrad(v.partialDerivative.get, computeGrad)

        })
      }
    }

  }

  def zipBackward(fn: (STen, STen) => Unit) = (this, fn)

  def accumulateGrad(
      incoming: STen,
      computeGrad: (STen, STen) => Unit
  ) = if (needsGrad) {
    computeGrad(incoming, partialDerivative.get)
  }

  import lamp.{scope => extractScope}
  def transpose[S: Sc](dim1: Int, dim2: Int) =
    new Transpose(extractScope, this, dim1, dim2).value
  def t[S: Sc] = new Transpose(extractScope, this).value
  def select[S: Sc](dim: Long, index: Long) =
    new Select(extractScope, this, dim = dim, index = index).value
  def indexSelect[S: Sc](dim: Long, index: Variable) =
    new IndexSelect(extractScope, this, dim = dim, index = index).value
  def argmax[S: Sc](dim: Long, keepDim: Boolean) =
    new ArgMax(extractScope, this, dim = dim, keepDim = keepDim).value
  def oneHot[S: Sc](numClasses: Int) =
    new OneHot(extractScope, this, numClasses).value
  def assign[S: Sc](other: Variable) =
    new Assign(extractScope, abandon = this, keep = other).value
  def maskFill[S: Sc](mask: Variable, fill: Double) =
    new MaskFill(extractScope, this, mask, fill).value
  def makeBooleanMask[S: Sc](q: Long) = new EqWhere(extractScope, this, q).value
  def cast[S: Sc](precision: FloatingPointPrecision) =
    new CastToPrecision(extractScope, this, precision).value
  def cat[S: Sc](other: Variable, dim: Long) =
    new Concatenate(extractScope, List(this, other), dim).value
  def +[S: Sc](other: Variable) = new Add(extractScope, this, other).value
  def +[S: Sc](other: Double) = new ConstAdd(extractScope, this, other).value
  def -[S: Sc](other: Variable) = new Minus(extractScope, this, other).value
  def *[S: Sc](other: Variable) = new Mult(extractScope, this, other).value
  def *[S: Sc](other: Double) = new ConstMult(extractScope, this, other).value
  def /[S: Sc](other: Variable) = new Div(extractScope, this, other).value
  def mm[S: Sc](other: Variable) = new MatMul(extractScope, this, other).value
  def bmm[S: Sc](other: Variable) =
    new BatchedMatMul(extractScope, this, other).value
  def relu[S: Sc] = new Relu(extractScope, this).value
  def swish1[S: Sc] = this * this.sigmoid
  def gelu[S: Sc] = new Gelu(extractScope, this).value
  def sigmoid[S: Sc] = new Sigmoid(extractScope, this).value
  def dropout[S: Sc](prob: Double, train: Boolean) =
    new Dropout(extractScope, this, prob, train).value
  def scatterAdd[S: Sc](index: Variable, dim: Int, maxIndex: Long) =
    new ScatterAdd(extractScope, this, index, dim, maxIndex).value
  def indexAdd[S: Sc](index: Variable, dim: Int, maxIndex: Long) =
    new IndexAdd(extractScope, this, index, dim, maxIndex).value
  def sum[S: Sc] = new Sum(extractScope, this, Nil, false).value
  def sum[S: Sc](dim: List[Int], keepDim: Boolean) =
    new Sum(extractScope, this, dim, keepDim).value
  def expandAs[S: Sc](other: STen) =
    new ExpandAs(extractScope, this, other).value
  def rowSum[S: Sc] = sum(List(1), true)
  def colSum[S: Sc] = sum(List(0), true)
  def exp[S: Sc] = new Exp(extractScope, this).value
  def log[S: Sc] = new Log(extractScope, this).value
  def log1p[S: Sc] = new Log1p(extractScope, this).value
  def sin[S: Sc] = new Sin(extractScope, this).value
  def cos[S: Sc] = new Cos(extractScope, this).value
  def tan[S: Sc] = new Tan(extractScope, this).value
  def tanh[S: Sc] = new Tanh(extractScope, this).value
  def atan[S: Sc] = new ArcTan(extractScope, this).value
  def pow[S: Sc](const: Double) = new PowConst(extractScope, this, const).value
  def pow[S: Sc](exponent: Variable) =
    new Pow(extractScope, this, exponent).value
  def euclideanDistance[S: Sc](b: Variable, dim: Int) =
    new EuclideanDistance(extractScope, this, b, dim).value
  def logSoftMax[S: Sc](dim: Int) =
    new LogSoftMax(extractScope, this, dim).value
  def crossEntropy[S: Sc](other: Variable) =
    ((this.*(other)).rowSum).*(-1d)
  def nllLoss[S: Sc](
      target: STen,
      weights: STen,
      reduction: Reduction = Mean,
      ignore: Long = -100L
  ) =
    new NllLoss(
      extractScope,
      this,
      target,
      weights,
      reduction,
      ignore
    ).value
  def mseLoss[S: Sc](
      target: STen,
      reduction: Reduction = Mean
  ) =
    new MseLoss(extractScope, this, target, reduction).value
  def l1Loss[S: Sc](
      target: STen,
      reduction: Reduction = Mean
  ) =
    new L1Loss(extractScope, this, target, reduction).value
  def squaredFrobenius[S: Sc] =
    new SquaredFrobeniusMatrixNorm(extractScope, this).value
  def mean[S: Sc](dim: List[Int]) =
    new Mean(extractScope, this, dim, true).value
  def mean[S: Sc](dim: List[Int], keepDim: Boolean) =
    new Mean(extractScope, this, dim, keepDim).value
  def variance[S: Sc](dim: List[Int]) =
    new Variance(extractScope, this, dim).value
  def normalize[S: Sc](dim: List[Int]) = {
    (this - this.mean(dim)) / ((this.variance(dim) + 1e-6).pow(0.5))
  }
  def view[S: Sc](shape: List[Long]) =
    new View(extractScope, this, shape.toArray).value
  def flatten[S: Sc] =
    new Flatten(extractScope, this, startDim = 0, endDim = -1).value
  def flatten[S: Sc](startDim: Int) =
    new Flatten(extractScope, this, startDim = startDim, endDim = -1).value
  def flatten[S: Sc](startDim: Int, endDim: Int) =
    new Flatten(extractScope, this, startDim = startDim, endDim = endDim).value
  def flattenLastDimensions[S: Sc](dims: Int) =
    new Flatten(
      extractScope,
      this,
      startDim = shape.size - dims,
      endDim = -1
    ).value
  def repeatInterleave[S: Sc](repeats: Variable, dim: Int) =
    new RepeatInterleave(extractScope, this, repeats, dim).value

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
          val children = n.op.toList.flatMap(_.params.map(_._1))
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
