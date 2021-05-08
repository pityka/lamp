package lamp.autograd
import java.{util => ju}
import lamp.FloatingPointPrecision
import lamp.Scope
import lamp.Sc
import lamp.STen
import lamp.Movable

/** Represents an operation in the computational graph
  *
  * ===Short outline of reverse autograd from scalar values===
  * `y = f1 o f2 o .. o fn`
  *
  * One of these subexpression (f_i) has value w2 and arguments `w1`.
  * We can write `dy/dw1 = dy/dw2 * dw2/dw1`.
  * `dw2/dw1` is the Jacobian of `f_i` at the current value of `w1`.
  * `dy/dw2` is the Jacobian of `y` wrt to `w2` at the current value of `w2`.
  *
  * The current value of `w1` and `w2` are computed in a forward pass.
  * The value `dy/dy` is 1 and from this `dy/dw2` is recursed in the backward pass.
  * The Jacobian function of `dw2/dw1` is computed symbolically and hard coded.
  *
  * The anonymous function which `Op`s must implement is `dy/dw2 => dy/dw2 * dw2/dw1`.
  * The argument of that function (`dy/dw2`) is coming down from the backward pass.
  * The `Op` must implement `dy/dw2 * dw2/dw1`.
  *
  * The shape of `dy/dw2` is the shape of the value of the operation (`dy/dw2`).
  * The shape of `dy/dw2 * dw2/dw1` is the shape of the parameter variable with respect which
  * the derivative is taken, i.e. `w1` since we are computing `dy/dw1`.
  *
  * ===How to implement an operation===
  * {{{
  * // Each concrete realization of the operation corresponds to an instance of an Op
  * // The Op instance holds handles to the input variables (here a, b), to be used in the backward pass
  * // The forward pass is effectively done in the constructor of the Op
  * // The backward pass is triggerd and orchestrated by [[lamp.autograd.Variable.backward]]
  * case class Mult(scope: Scope, a: Variable, b: Variable) extends Op {
  *
  * // List all parameters which support partial derivatives, here both a and b
  * val params = List(
  *  // partial derivative of the first argument
  *  a.zipBackward { (p, out) =>
  *   // p is the incoming partial derivative, out is where the result is accumated into
  *   // Intermediate tensors are released due to the enclosing Scope.root
  *   Scope.root { implicit scope => out += (p * b.value).unbroadcast(a.sizes) }
  *   },
  *  // partial derivative of the second argument ..
  *  b.zipBackward { (p, out) =>
  *   Scope.root { implicit scope => out += (p * a.value).unbroadcast(b.sizes) }
  *
  *   }
  * )
  * //The value of this operation, i.e. the forward pass
  * val value = Variable(this, a.value.*(b.value)(scope))(scope)
  *
  * }
  * }}}
  * @see [[https://en.wikipedia.org/wiki/Automatic_differentiation#Reverse_accumulation]]
  * @see [[http://www.cs.cmu.edu/~wcohen/10-605/notes/autodiff.pdf]]
  */
trait Op {

  /** The value of this operation */
  val value: Variable

  /** Implementation of the backward pass
    *
    * A list of input variables paired up with an anonymous function computing the respective partial
    * derivative. With the notation in the documentation of the trait [[lamp.autograd.Op]]:
    * `dy/dw2 => dy/dw2 * dw2/dw1`. The first argument of the anonymous function is the incoming
    * partial derivative (`dy/dw2`), the second argument is the output tensor into which the
    * result (`dy/dw2 * dw2/dw1`) is accumulated (added).
    *
    * If the operation does not support computing the partial derivative for some of its arguments, then
    * do not include that argument in this list.
    *
    * @see The documentation on the trait [[lamp.autograd.Op]] for more details and example.
    */
  val params: List[(Variable, (STen, STen) => Unit)]
}

object Variable {
  def apply(op: Op, value: STen)(implicit
      scope: Scope
  ): Variable =
    VariableNonConstant(
      op,
      value,
      STen.zerosLike(value)(scope)
    )

  /** Same as [[lamp.autograd.Variable.stack]] */
  def concatenateAddNewDim(inputs: Seq[Variable])(implicit scope: Scope) =
    new Stack(scope, inputs, 0).value

  /** Concatenates the given tensor along a newly inserted dimension */
  def stack(inputs: Seq[Variable], dim: Int)(implicit scope: Scope) =
    new Stack(scope, inputs, dim).value

  /** Concatenates the given tensor along the given dimension */
  def cat(inputs: Seq[Variable], dim: Long)(implicit scope: Scope) =
    new Concatenate(scope, inputs, dim).value
}

/** A variable whose parent and partial derivatives are empty */
case class ConstantWithoutGrad(
    value: STen
) extends Constant {
  val partialDerivative = None
}

/** A variable whose parent is empty but whose partial derivative is defined */
case class ConstantWithGrad(
    value: STen,
    pd: STen
) extends Constant {
  val partialDerivative = Some(pd)
}

/** A variable whose parent is empty */
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

/** A variable whose parent is not empty, neither its partial derivative */
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

/** A value of a tensor valued function, a vertex in the computational graph.
  *
  * A Variable may be constant, i.e. depends on no other Variables.
  * Constant variables may or may not need their partial derivatives computed.
  */
sealed trait Variable {

  /** The parent operation of this value in the computational graph. Empty for constants. */
  def op: Option[Op]

  /** The actual tensor value of this Variable. */
  def value: STen

  /** The partial derivative, or a placeholder tensor for the partial derivative.
    *
    * Returns empty iff this Variable needs no gradient computation. Otherwise a placeholder tensor
    * is allocated upfront when the Variable is allocated.
    */
  def partialDerivative: Option[STen]

  /** Returns true if [[lamp.autograd.Variable.partialDerivative]] is defined. */
  def needsGrad: Boolean = partialDerivative.isDefined

  override def toString =
    s"Var(value=$value,needsGrad=$needsGrad)"

  /** Returns the tensor options of its value. */
  def options[S: Sc] = value.options

  /** Returns the shape of its value. */
  val sizes = value.sizes.toList

  /** Returns the shape of its value. */
  def shape = sizes

  /** Returns unique, stable and random UUID. */
  val id = ju.UUID.randomUUID()

  /** Returns an other Variable wrapping the same value tensor, without any parent and with `needsGrad=false`. */
  def detached = const(value)

  /** Returns an other Variable wrapping the same value tensor, without any parent and with `needsGrad=true`. */
  def withGrad[S: Sc] = param(value)

  /** In place zeros out the partial derivative */
  def zeroGrad() = {
    partialDerivative.foreach { t => t.zero_() }
  }

  /** Returns the Wengert list */
  lazy val wengert = Autograd.topologicalSort(this)

  /** Runs the backpropagation algorithm starting from this value
    *
    * Only meaningful if this is scalar i.e. the number of elements in the value tensor is 1.
    */
  def backprop(): Unit = {
    if (partialDerivative.isDefined) {
      partialDerivative.get.fill_(1d)
      wengert.foreach { v =>
        v.op.foreach(_.params.foreach { case (v1, computeGrad) =>
          v1.accumulateGrad(v.partialDerivative.get, computeGrad)

        })
      }
    }

  }

  /** Returns a pair of this instance and the supplied function */
  def zipBackward(fn: (STen, STen) => Unit) = (this, fn)

  private def accumulateGrad(
      incoming: STen,
      computeGrad: (STen, STen) => Unit
  ) = if (needsGrad) {
    computeGrad(incoming, partialDerivative.get)
  }

  import lamp.{scope => extractScope}

  /** Returns a new variable with the respective dimensions transposed. */
  def transpose[S: Sc](dim1: Int, dim2: Int) =
    new Transpose(extractScope, this, dim1, dim2).value

  /** Returns a new variable with the first two dimensions transposed. */
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
  def maskSelect[S: Sc](mask: Variable) =
    new MaskSelect(extractScope, this, mask).value
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
  def leakyRelu[S: Sc](negativeSlope: Double) =
    new LeakyRelu(extractScope, this, negativeSlope).value
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
  def binaryCrossEntropyWithLogitsLoss[S: Sc](
      target: STen,
      posWeights: Option[STen] = None,
      reduction: Reduction = Mean
  ) =
    new BinaryCrossEntropyWithLogitsLoss(
      extractScope,
      this,
      target,
      posWeights,
      reduction
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
  def reshape[S: Sc](shape: List[Long]) =
    new Reshape(extractScope, this, shape.toArray).value
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
