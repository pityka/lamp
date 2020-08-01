package lamp.autograd
import aten.{Tensor, ATen}
import java.{util => ju}
import aten.TensorOptions
import scala.collection.mutable
import lamp.FloatingPointPrecision
import lamp.syntax

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

class AllocatedVariablePool {
  private val buffer0 = scala.collection.mutable.ArrayBuffer[Variable]()
  private val buffer1 = scala.collection.mutable.ArrayBuffer[Tensor]()

  private val leased =
    scala.collection.mutable.HashSet[(List[Long], Tensor)]()
  private val leasables =
    scala.collection.mutable.AnyRefMap[List[Long], List[Tensor]]()

  def askForLease(shape: List[Long], tOpt: TensorOptions) = {
    leasables.get(shape) match {
      case None | Some(Nil) =>
        val t = ATen.zeros(shape.toArray, tOpt)
        leased += ((shape, t))
        t
      case Some(x :: xs) =>
        leasables.update(shape, xs)
        leased += ((shape, x))
        ATen.zero_(x)
        x

    }
  }

  def returnLease(shape: List[Long], tensor: Tensor) = {
    leased -= ((shape, tensor))
    leasables.get(shape) match {
      case None    => leasables.update(shape, List(tensor))
      case Some(l) => leasables.update(shape, tensor :: l)
    }
  }

  def append(v: Variable) = buffer0.append(v)
  def appendTensor(t: Tensor) = buffer1.append(t)
  def releaseAll() = {
    val buffer = mutable.ArrayBuffer[Tensor]()
    buffer0.foreach { variable =>
      buffer.append(variable.value)
      variable.partialDerivative.foreach { pd =>
        returnLease(variable.shape, pd)
      }
    }
    buffer1.foreach { t => buffer.append(t) }
    Tensor.releaseAll(buffer.distinct.toArray)
    buffer0.clear()
    buffer1.clear()
  }
}

// Variable takes ownership of the value: Tensor
// therefore it must be the sole owner
case class Variable(
    op: Op,
    value: Tensor,
    pool: AllocatedVariablePool,
    needsGrad: Boolean = true
) {

  override def toString =
    s"Var(shape=$shape,value=$value,needsGrad=$needsGrad)"

  val options = value.options

  var partialDerivative: Option[Tensor] = None

  val sizes = value.sizes.toList

  def shape = sizes

  val id = ju.UUID.randomUUID()

  def releaseAll(): Unit = {
    pool.releaseAll

  }
  def releasable = {
    pool.append(this)
    this
  }
  def releaseWith(t: Tensor*) = {
    t.foreach { t => pool.appendTensor(t) }
    this
  }
  def releaseWithVariable(t: Variable*) = {
    t.foreach { t => pool.append(t) }
    this
  }
  def needsNoGrad = copy(needsGrad = false)
  def detached = const(value)(pool).releasable
  def zeroGrad() = {
    partialDerivative.foreach { t => ATen.zero_(t) }
  }

  lazy val wengert = Autograd.topologicalSort(this)

  def backprop(): Unit = {
    if (partialDerivative.isEmpty) {
      partialDerivative = Some(
        ATen.ones_like(value, value.options)
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
      partialDerivative = Some(pool.askForLease(shape, options))
    }
    computeGrad(incoming, partialDerivative.get)
  }

  def t = Transpose(this).value
  def transpose(dim1: Int, dim2: Int) = Transpose(this, dim1, dim2).value
  def select(dim: Long, index: Long) =
    Select(this, dim = dim, index = index).value
  def indexSelect(dim: Long, index: Variable) =
    IndexSelect(this, dim = dim, index = index).value
  def argmax(dim: Long, keepDim: Boolean) =
    ArgMax(this, dim = dim, keepDim = keepDim).value
  def oneHot(numClasses: Int) =
    OneHot(this, numClasses).value
  def assign(other: Variable) = Assign(abandon = this, keep = other).value
  def maskFill(mask: Variable, fill: Double) = MaskFill(this, mask, fill).value
  def makeBooleanMask(q: Long) = EqWhere(this, q).value
  def cast(precision: FloatingPointPrecision) =
    CastToPrecision(this, precision).value
  def cat(other: Variable, dim: Long) =
    Concatenate(List(this, other), dim).value
  def +(other: Variable) = Add(this, other).value
  def +(other: Double) = ConstAdd(this, other).value
  def -(other: Variable) = Minus(this, other).value
  def *(other: Variable) = Mult(this, other).value
  def *(other: Double) = ConstMult(this, other).value
  def /(other: Variable) = Div(this, other).value
  def mm(other: Variable) = MatMul(this, other).value
  def bmm(other: Variable) = BatchedMatMul(this, other).value
  def relu = Relu(this).value
  def gelu = Gelu(this).value
  def sigmoid = Sigmoid(this).value
  def dropout(prob: Double, train: Boolean) = Dropout(this, prob, train).value
  def sum = Sum(this).value
  def rowSum = RowSum(this).value
  def colSum = ColSum(this).value
  def exp = Exp(this).value
  def log = Log(this).value
  def sin = Sin(this).value
  def cos = Cos(this).value
  def tan = Tan(this).value
  def tanh = Tanh(this).value
  def atan = ArcTan(this).value
  def pow(const: Double) = PowConst(this, const).value
  def logSoftMax(dim: Int) = LogSoftMax(this, dim).value
  def crossEntropy(other: Variable) =
    ((this.*(other)).rowSum).*(-1d)
  def nllLoss(
      target: Tensor,
      numClasses: Int,
      weights: Tensor,
      reduction: Reduction = Mean,
      ignore: Long = -100L
  ) =
    NllLoss(this, target, weights, numClasses, reduction, ignore).value
  def mseLoss(
      target: Tensor,
      reduction: Reduction = Mean
  ) =
    MseLoss(this, target, reduction).value
  def l1Loss(
      target: Tensor,
      reduction: Reduction = Mean
  ) =
    L1Loss(this, target, reduction).value
  def squaredFrobenius = SquaredFrobeniusMatrixNorm(this).value
  def mean(dim: List[Int]) = Mean(this, dim).value
  def view(shape: List[Int]) = View(this, shape.map(_.toLong).toArray).value
  def flattenLastDimensions(dims: Int) = FlattenLastDimensions(this, dims).value

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
