package lamp.autograd
// import lamp.FloatingPointPrecision
import lamp.Scope
import lamp.STen
import lamp.Movable
import lamp.STenOptions
import lamp.Device
import lamp.FloatingPointPrecision

/** Represents an operation in the computational graph
  *
  * ===Short outline of reverse autograd from scalar values===
  * `y = f1 o f2 o .. o fn`
  *
  * One of these subexpression (f_i) has value w2 and arguments `w1`. We can
  * write `dy/dw1 = dy/dw2 * dw2/dw1`. `dw2/dw1` is the Jacobian of `f_i` at the
  * current value of `w1`. `dy/dw2` is the Jacobian of `y` wrt to `w2` at the
  * current value of `w2`.
  *
  * The current value of `w1` and `w2` are computed in a forward pass. The value
  * `dy/dy` is 1 and from this `dy/dw2` is recursed in the backward pass. The
  * Jacobian function of `dw2/dw1` is computed symbolically and hard coded.
  *
  * The anonymous function which `Op`s must implement is `dy/dw2 => dy/dw2 *
  * dw2/dw1`. The argument of that function (`dy/dw2`) is coming down from the
  * backward pass. The `Op` must implement `dy/dw2 * dw2/dw1`.
  *
  * The shape of `dy/dw2` is the shape of the value of the operation (`dy/dw2`).
  * The shape of `dy/dw2 * dw2/dw1` is the shape of the parameter variable with
  * respect which the derivative is taken, i.e. `w1` since we are computing
  * `dy/dw1`.
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
  *   // partial derivative of the first argument
  *   a.zipBackward { (p, out) =>
  *   // p is the incoming partial derivative, out is where the result is accumated into
  *   // Intermediate tensors are released due to the enclosing Scope.root
  *   Scope.root { implicit scope => out += (p * b.value).unbroadcast(a.sizes) }
  *   },
  *   // partial derivative of the second argument ..
  *   b.zipBackward { (p, out) =>
  *   Scope.root { implicit scope => out += (p * a.value).unbroadcast(b.sizes) }
  *
  *   }
  * )
  * //The value of this operation, i.e. the forward pass
  * val value = Variable(this, a.value.*(b.value)(scope))(scope)
  *
  * }
  * }}}
  * @see
  *   [[https://en.wikipedia.org/wiki/Automatic_differentiation#Reverse_accumulation]]
  * @see
  *   [[http://www.cs.cmu.edu/~wcohen/10-605/notes/autodiff.pdf]]
  */
trait Op {

  /** The value of this operation */
  val value: Variable

  /** Implementation of the backward pass
    *
    * A list of input variables paired up with an anonymous function computing
    * the respective partial derivative. With the notation in the documentation
    * of the trait [[lamp.autograd.Op]]: `dy/dw2 => dy/dw2 * dw2/dw1`. The first
    * argument of the anonymous function is the incoming partial derivative
    * (`dy/dw2`), the second argument is the output tensor into which the result
    * (`dy/dw2 * dw2/dw1`) is accumulated (added).
    *
    * If the operation does not support computing the partial derivative for
    * some of its arguments, then do not include that argument in this list.
    *
    * @see
    *   The documentation on the trait [[lamp.autograd.Op]] for more details and
    *   example.
    */
  val params: List[Either[ParamHelper, Variable]]
  val joinedBackward: Option[Scope => JoinedGradHelper => Unit] = None
  val cacheHint: Boolean = false
}
case class Grad(p: STen, out: STen, fw: ForwardCache)
object Grad {
  def unapply(grad: Grad) = Some((grad.p, grad.out))
}

case class ParamHelper(variable: Variable, accumulateGrad: Grad => Unit)

object Variable {
  def persist(op: Op, makeValue: Scope => ForwardCache => STen): Variable =
    new LazyVariableNonConstant(
      op,
      { (scope: Scope) => forward =>
        makeValue(scope)(forward) -> Nil
      },
      true
    )
  def apply(op: Op, makeValue: Scope => ForwardCache => STen): Variable =
    new LazyVariableNonConstant(
      op,
      { (scope: Scope) => forward =>
        makeValue(scope)(forward) -> Nil
      },
      false
    )
  def withAux(
      op: Op,
      makeValue: Scope => ForwardCache => (STen, List[STen])
  ): Variable =
    new LazyVariableNonConstant(
      op,
      { (scope: Scope) => forward =>
        makeValue(scope)(forward)

      },
      false
    )

  /** Same as [[lamp.autograd.Variable.stack]] */
  def concatenateAddNewDim(inputs: Seq[Variable]) =
    new Stack(inputs, 0).value

  /** Concatenates the given tensor along a newly inserted dimension */
  def stack(inputs: Seq[Variable], dim: Int) =
    new Stack(inputs, dim).value

  /** Concatenates the given tensor along the given dimension */
  def cat(inputs: Seq[Variable], dim: Long) =
    new Concatenate(inputs, dim).value

  def where(condition: STen, trueBranch: Variable, falseBranch: Variable) =
    new Where(condition, trueBranch, falseBranch).value

  // def sparseFromValueAndIndex(values: Variable, indices: STen, dim: Seq[Long]) =
  // new SparseFromValueAndIndex(values, indices, dim).value
}

/** A variable whose parent and partial derivatives are empty */
final class ConstantWithoutGrad(
    value1: STen
) extends Constant {
  protected def valueAux(forward: ForwardCache)(implicit
      scope: Scope
  ): (STen, List[STen]) = (value1, Nil)

  def constantValue: STen = value1

  /** Returns an other Variable wrapping the same value tensor eligible for
    * gradient calculation
    */
  def withGrad = param(value1)

  override def toString =
    s"Constant(value=$value1) ${System.identityHashCode(this)}"
}

/** A variable whose parent is empty but whose partial derivative is defined */
final class ConstantWithGrad(
    value1: STen
) extends Constant {
  protected def valueAux(forward: ForwardCache)(implicit
      scope: Scope
  ): (STen, List[STen]) = (value1, Nil)
  def constantValue = value1
  override def toString =
    s"Param(value=$value1)"
}

/** A variable whose parent is empty */
sealed trait Constant extends Variable {
  final def op = None
  def constantValue: STen

  def allocatePartialDerivative(implicit scope: Scope) =
    STen.zerosLike(constantValue)

}

object Constant {
  implicit val movable: Movable[Constant] =
    Movable.nonEmpty[Constant] {
      case p: ConstantWithGrad    => List(p.constantValue.value)
      case p: ConstantWithoutGrad => List(p.constantValue.value)
    }

}

trait ForwardCache {
  def getValues(v: LazyVariableNonConstant): Option[(STen, List[STen])]
  def getDescriptors(
      v: LazyVariableNonConstant
  ): (List[Long], Byte, lamp.Device)
  def offer(v: LazyVariableNonConstant, value: STen, aux: List[STen]): Unit
  def scope: Scope
  def allValues: Seq[STen]
}

object ForwardCache {
  def selective(implicit sc: Scope) = Autograd.BackpropForwardCache.empty(
    Autograd.CacheStrategy.SelectivelyCache,
    Nil
  )
  def caching(implicit sc: Scope) =
    Autograd.BackpropForwardCache.empty(Autograd.CacheStrategy.AlwaysCache, Nil)
}

/** A variable whose parent is not empty, neither its partial derivative */
final class LazyVariableNonConstant(
    op1: Op,
    makeValue: Scope => ForwardCache => (STen, List[STen]),
    cacheHint0: Boolean
) extends Variable {

  def cacheHint = cacheHint0

  def valueAux(forward: ForwardCache)(implicit scope: Scope) = {
    def make =
      makeValue(scope)(forward)

    forward.getValues(this) match {
      case None =>
        val (v, aux) = make
        forward.offer(this, v, aux)
        (v, aux)
      case Some(pair) =>
        pair
    }
  }

  override def toString =
    s"Lazy(${System.identityHashCode(this)} ${op1.getClass()})"

  val op = Some(op1)
}

case class JoinedGradHelper(
    p: STen,
    partialDerivatives: scala.collection.mutable.Map[Variable, STen],
    forwardCache: ForwardCache
) {
  def partialDerivative(v: Variable)(scope: Scope) =
    partialDerivatives.get(v) match {
      case Some(v) => v
      case None =>
        val pd = Autograd.allocatePartialDerivative(v, forwardCache)(scope)
        partialDerivatives.update(v, pd)
        pd
    }

}

sealed trait Variable {

  /** Returns a new variable with the respective dimensions transposed. */
  def transpose(dim1: Int, dim2: Int) =
    new Transpose(this, dim1, dim2).value

  /** Returns a new variable with the first two dimensions transposed. */
  def t = new Transpose(this).value
  def select(dim: Long, index: Long) =
    new Select(this, dim = dim, index = index).value
  def slice(dim: Long, start: Long, end: Long, step: Long) =
    new Slice(
      this,
      dim = dim,
      start = start,
      end = end,
      step = step
    ).value
  def indexSelect(dim: Long, index: STen) =
    new IndexSelect(this, dim = dim, index = index).value
  def argmax(dim: Long, keepDim: Boolean) =
    new ArgMax(this, dim = dim, keepDim = keepDim).value
  def oneHot(numClasses: Int) =
    new OneHot(this, numClasses).value

  /** x = a.assign(b) x's value is b, x's grad is 1 wrt b and 0 wrt to a
    */
  def assign(other: Variable) =
    new Assign(abandon = this, keep = other).value
  def maskFill(mask: STen, fill: Double) =
    new MaskFill(this, mask, fill).value
  def maskSelect(mask: Variable) =
    new MaskSelect(this, mask).value
  def makeBooleanMask(q: Long) = new EqWhere(this, q).value
  def cast(precision: FloatingPointPrecision) =
    new CastToPrecision(this, precision).value
  def cat(other: Variable, dim: Long) =
    new Concatenate(List(this, other), dim).value
  def +(other: Variable) = new Add(this, other).value
  def +(other: Double) = new ConstAdd(this, other).value
  def -(other: Variable) = new Minus(this, other).value
  def *(other: Variable) = new Mult(this, other).value
  def *(other: Double) = new ConstMult(this, other).value
  def /(other: Variable) = new Div(this, other).value
  def cross(other: Variable, dim: Int) =
    new Cross(this, other, dim).value
  def mm(other: Variable) = new MatMul(this, other).value
  def bmm(other: Variable) =
    new BatchedMatMul(this, other).value
  def relu = new Relu(this).value
  def leakyRelu(negativeSlope: Double) =
    new LeakyRelu(this, negativeSlope).value
  def swish1 = this * this.sigmoid
  def gelu = new Gelu(this).value
  def sigmoid = new Sigmoid(this).value
  def hardSwish = new HardSwish(this).value
  def dropout(prob: Double, train: Boolean) =
    new Dropout(this, prob, train).value
  def scatterAdd(index: STen, dim: Int, maxIndex: Long) =
    new ScatterAdd(this, index, dim, maxIndex).value.persist
  def indexAdd(index: STen, dim: Int, maxIndex: Long) =
    new IndexAdd(this, index, dim, maxIndex).value.persist
  def indexAddFromSource(index: STen, dim: Int, source: Variable) =
    new IndexAddToTarget(this, source, index, dim).value.persist
  def indexFill(index: STen, dim: Int, fillValue: Double) =
    new IndexFill(this, dim, index, fillValue).value.persist

  def sum = new Sum(this, Nil, false).value
  def sum(dim: List[Int], keepDim: Boolean) =
    new Sum(this, dim, keepDim).value
  def norm2(dim: List[Int]) =
    new Norm2(this, dim, false).value
  def norm2(dim: List[Int], keepDim: Boolean) =
    new Norm2(this, dim, keepDim).value
  def expandAs(other: STen) =
    new ExpandAs(this, other).value
  def expand(shape: List[Long]) =
    new Expand(this, shape).value
  def rowSum = sum(List(1), true)
  def colSum = sum(List(0), true)
  def exp = new Exp(this).value
  def logdet = new LogDet(this).value
  def log = new Log(this).value
  def log1p = new Log1p(this).value
  def softplus(beta: Double, threshold: Double) =
    new Softplus(this, beta, threshold).value
  def sin = new Sin(this).value
  def cos = new Cos(this).value
  def tan = new Tan(this).value
  def tanh = new Tanh(this).value
  def atan = new ArcTan(this).value
  def pow(const: Double) = new PowConst(this, const).value
  def pow(exponent: Variable) =
    new Pow(this, exponent).value
  def euclideanDistance(b: Variable, dim: Int) =
    new EuclideanDistance(this, b, dim).value
  def logSoftMax(dim: Int) =
    new LogSoftMax(this, dim).value
  def crossEntropy(other: Variable) =
    ((this.*(other)).rowSum).*(-1d)
  def nllLoss(
      target: STen,
      weights: STen,
      reduction: Reduction = Mean,
      ignore: Long = -100L
  ) =
    new NllLoss(
      this,
      target,
      weights,
      reduction,
      ignore
    ).value
  def binaryCrossEntropyWithLogitsLoss(
      target: STen,
      posWeights: Option[STen] = None,
      reduction: Reduction = Mean
  ) =
    new BinaryCrossEntropyWithLogitsLoss(
      this,
      target,
      posWeights,
      reduction
    ).value

  def mseLoss(
      target: STen,
      reduction: Reduction = Mean
  ) =
    new MseLoss(this, target, reduction).value
  def smoothL1Loss(
      target: STen,
      reduction: Reduction = Mean,
      beta: Double = 1.0
  ) =
    new SmoothL1Loss(this, target, reduction, beta).value
  def mean(dim: List[Int]) =
    new Mean(this, dim, true).value
  def mean(dim: List[Int], keepDim: Boolean) =
    new Mean(this, dim, keepDim).value
  def variance(dim: List[Int]) =
    new Variance(this, dim).value
  def normalize(dim: List[Int], eps: Double) = {
    (this - this.mean(dim)) / ((this.variance(dim) + eps).pow(0.5))
  }
  def view(shape: List[Long]) =
    new View(this, shape).value
  def reshape(shape: List[Long]) =
    new Reshape(this, shape).value
  def flatten =
    new Flatten(this, startDim = 0, endDim = -1).value
  def flatten(startDim: Int) =
    new Flatten(this, startDim = startDim, endDim = -1).value
  def flatten(startDim: Int, endDim: Int) =
    new Flatten(this, startDim = startDim, endDim = endDim).value
  def flattenLastDimensions(dims: Int) =
    new FlattenLastDimensions(
      this,
      dims
    ).value
  def repeatInterleave(repeats: STen, dim: Int) =
    new RepeatInterleave(this, repeats, dim).value
  def toDense = new ToDense(this).value
  def diag(diagonal: Long) = new Diag(this, diagonal).value
  def inv = new Inv(this).value
  def pinv(rcond: Double = 1e-5) =
    new PInv(this, rcond).value.persist
  def choleskyLower =
    new Cholesky(this).value.persist
  def choleskySolve(factor: Variable, upper: Boolean = false) =
    new CholeskySolve(
      b = this,
      factor = factor,
      upper = upper
    ).value.persist
  def minimum(other: Variable) =
    new ElementWiseMinimum(this, other).value
  def maximum(other: Variable) =
    new ElementWiseMaximum(this, other).value
  def clamp(min: Variable, max: Variable) =
    (this.minimum(max)).maximum(min)
  // def debug(fun: (STen, Boolean, Boolean) => Unit) =
  //   new Debug( this, fun).value

  def persist = new Persist(this).value

  /** The parent operation of this value in the computational graph. Empty for
    * constants.
    */
  def op: Option[Op]

  /** The actual tensor value of this Variable. */
  protected def valueAux(forward: ForwardCache)(implicit
      scope: Scope
  ): (STen, List[STen])

  def eval(implicit scope: Scope): STen =
    evalAux._1

  private def evalAux(implicit scope: Scope): (STen, List[STen]) = {
    val fw = ForwardCache.caching
    val (a, v) = forwardAux(fw)
    (a.cloneTensor, v.map(_.cloneTensor))
  }

  def shape(implicit fw: ForwardCache) = this match {
    case t: Constant                => t.constantValue.shape
    case t: LazyVariableNonConstant => t.forward.shape
  }
  def device(implicit fw: ForwardCache) = this match {
    case t: Constant                => t.constantValue.device
    case t: LazyVariableNonConstant => t.forward.device
  }
  def scalarTypeByte(implicit fw: ForwardCache) = this match {
    case t: Constant                => t.constantValue.scalarTypeByte
    case t: LazyVariableNonConstant => t.forward.scalarTypeByte
  }

  def options(implicit scope: Scope, fw: ForwardCache) =
    STenOptions.fromScalarType(scalarTypeByte, device)

  def forward(implicit
      fw: ForwardCache
  ): STen = forwardAux(fw)._1

  private[autograd] def forwardAux(implicit
      fw: ForwardCache
  ): (STen, List[STen]) =
    valueAux(forward = fw)(fw.scope)

  /** Returns a pair of this instance and the supplied function */
  private[autograd] def zipBackward(
      fn: Grad => Unit
  ): Either[ParamHelper, Variable] =
    Left(ParamHelper(this, fn))
  private[autograd] def missingBackward: Either[ParamHelper, Variable] = Right(
    this
  )
}

// /** A value of a tensor valued function, a vertex in the computational graph.
//   *
//   * A Variable may be constant, i.e. depends on no other Variables. Constant
//   * variables may or may not need their partial derivatives computed.
//   */
// sealed trait MaybeNonDifferentiableVariable {

//   private[autograd] def missingBackward : Either[ParamHelper,MaybeNonDifferentiableVariable] = Right(this)

//   // def toDoubleArray =
//   //   Scope.root(scope => value(forward = true)(scope).toDoubleArray)
//   // def toLongArray =
//   //   Scope.root(scope => value(forward = true)(scope).toLongArray)
// }

object Autograd {

  //   /** Returns the value and a newly allocated zero tensor of the same shape
  //   */
  // private[autograd] def allocatePartialDerivative(forwardCache: ForwardCache)(scope: Scope): (STen) = {

  // }

  private[autograd] def allocatePartialDerivative(
      v: Variable,
      forwardCache: ForwardCache
  )(scope: Scope): STen = v match {
    case t: Constant => STen.zerosLike(t.constantValue)(scope)
    case t: LazyVariableNonConstant =>
      val (size, tpe, dev) = forwardCache.getDescriptors(t)
      STen.zeros(
        size,
        STenOptions.fromScalarType(tpe, dev)(scope)
      )(scope)
  }

  sealed trait CacheStrategy
  object CacheStrategy {
    case object NeverCache extends CacheStrategy
    case object AlwaysCache extends CacheStrategy

    /** Cache if either:
      *   - node degree is 2 or more
      *   - node definition is marked for caching
      *   - node application is marekd for caching with a call to .persist
      */
    case object SelectivelyCache extends CacheStrategy
  }
  import CacheStrategy._

  class BackpropForwardCache(
      cacheStrategy: CacheStrategy,
      values: scala.collection.mutable.Map[
        LazyVariableNonConstant,
        (STen, List[STen])
      ],
      descriptors: scala.collection.mutable.Map[
        LazyVariableNonConstant,
        (List[Long], Byte, lamp.Device)
      ],
      cacheVariables: List[Variable]
  )(implicit sc: Scope)
      extends ForwardCache {
    def allValues: Seq[STen] = values.values.toVector.flatMap { case (a, b) =>
      a :: b
    }
    val scope = sc
    def exists(value: STen) = values.exists(_._2._1 == value)
    def remove(v: LazyVariableNonConstant) = values.remove(v)
    def withCacheStrategy(hint: CacheStrategy): BackpropForwardCache =
      new BackpropForwardCache(hint, values, descriptors, cacheVariables)
    def getValues(v: LazyVariableNonConstant): Option[(STen, List[STen])] = {
      values.get(v)

    }
    def getDescriptors(
        v: LazyVariableNonConstant
    ): (List[Long], Byte, Device) = descriptors(v)

    private def cache(
        v: LazyVariableNonConstant,
        value: STen,
        aux: List[STen]
    ) = {
      assert(!values.contains(v))
      values.update(v, (value, aux))
    }
    def offer(
        v: LazyVariableNonConstant,
        value: STen,
        aux: List[STen]
    ): Unit = {
      val size = value.shape
      val tpe = value.scalarTypeByte
      val dev = value.device
      descriptors.update(v, (size, tpe, dev))

      if (cacheVariables.contains(v)) cache(v, value, aux)
      else
        cacheStrategy match {
          case NeverCache => ()
          case AlwaysCache =>
            cache(v, value, aux)
          case SelectivelyCache =>
            if (
              v.op.exists(_.cacheHint) || v.cacheHint || v.op.exists(
                _.params.size > 1
              )
            ) cache(v, value, aux)
        }
    }

  }
  object BackpropForwardCache {
    def empty(strategy: CacheStrategy, cacheVariables: List[Variable])(implicit
        scope: Scope
    ) =
      new BackpropForwardCache(
        strategy,
        scala.collection.mutable.Map(),
        scala.collection.mutable.Map(),
        cacheVariables
      )
    def simple(implicit scope: Scope) = empty(AlwaysCache, Nil)
  }

  def graphMemoryAllocationReport(
      root: Variable,
      fw: ForwardCache,
      partialDerivatives: Map[Variable, STen]
  ) = {
    var parameterTensorCount = 0L
    var parameterTensorStorage = 0L
    var constantTensorCount = 0L
    var constantTensorStorage = 0L
    var intermediateTensorCount = 0L
    var intermediateTensorStorage = 0L
    var pdTensorCount = 0L
    var pdTensorStorage = 0L
    val wengert = Autograd.topologicalSort(root)
    wengert
      .foreach {
        case t: ConstantWithGrad =>
          parameterTensorCount += 1
          parameterTensorStorage += t.constantValue.numBytes
        case t: ConstantWithoutGrad =>
          constantTensorCount += 1
          constantTensorStorage += t.constantValue.numBytes

        case _ =>
          ()
      }
    fw.allValues.distinct.foreach { sten =>
      intermediateTensorCount += 1
      intermediateTensorStorage += sten.numBytes
    }
    partialDerivatives.values.toList.distinct.foreach { case sten =>
      pdTensorCount += 1
      pdTensorStorage += sten.numBytes
    }
    GraphMemoryAllocationReport(
      parameterTensorCount,
      parameterTensorStorage,
      constantTensorCount,
      constantTensorStorage,
      intermediateTensorCount,
      intermediateTensorStorage,
      pdTensorCount,
      pdTensorStorage
    )
  }

  /** Runs the backpropagation algorithm starting from root
    *
    * Only meaningful if root is scalar i.e. the number of elements in the value
    * tensor is 1.
    */
  def backprop(
      root: Variable,
      partialDerivatives: Map[Variable, STen],
      forwardCache: BackpropForwardCache,
      printZeroGradients: Boolean
  )(scope: Scope) = {
    val wengert = Autograd.topologicalSort(root)
    val mutable =
      scala.collection.mutable.HashMap(partialDerivatives.toList: _*)

    def partialDerivative(v: Variable) = mutable.get(v) match {
      case Some(v) => v
      case None =>
        val pd = allocatePartialDerivative(v, forwardCache)(scope)
        mutable.update(v, pd)
        assert(mutable.contains(v))
        pd
    }

    partialDerivative(root).fill_(1d)

    wengert.zipWithIndex.foreach {
      case (v: Variable, wIdx) =>
        v.op.foreach { operator =>
          if (operator.joinedBackward.isEmpty) {
            operator.params.foreach {
              case Right(_) => ()
              case Left(ParamHelper(parentalVariable, accumulateGrad)) =>
                val needsGrad = parentalVariable match {
                  case _: ConstantWithoutGrad => false
                  case _                      => true
                }
                if (needsGrad) {
                  val pd = partialDerivative(parentalVariable)
                  accumulateGrad(
                    Grad(p = mutable(v), out = pd, fw = forwardCache)
                  )

                }

            }
          } else {
            operator.joinedBackward.get(scope)(
              JoinedGradHelper(mutable(v), mutable, forwardCache)
            )
          }

        }
        // look ahead in the wengert list and if v1 is not listed as an input then
        //  release v1.pd
        val pdNotNeeded = wengert
          .drop(wIdx)
          .forall(
            _.op.forall(op =>
              op.params
                .map(_.fold(_.variable, id => id))
                .forall(parental => parental != v)
            )
          )
        if (pdNotNeeded && printZeroGradients) {
          Scope.root { implicit scope =>
            val pd = mutable.get(v)
            pd.foreach { pd =>
              val norm = pd
                .view(-1)
                .norm2(List(0), false)
                .toDevice(lamp.CPU)
                .toDoubleArray(0)
              if (norm < 1e-5) {
                println(s"Near zero gradient of $v $norm")
              }
            }
          }
        }
        val canRelease = v match {
          case _: LazyVariableNonConstant => true
          case _                          => false
        }
        if (pdNotNeeded && canRelease) {
          val pd = mutable(v)

          mutable.remove(v)
          scope.release(pd.value)
          // release unneeded forwards here
          v match {
            case t: LazyVariableNonConstant =>
              forwardCache.getValues(t) match {
                case None =>
                case Some((va, aux)) =>
                  forwardCache.remove(t)
                  if (!forwardCache.exists(va)) {
                    scope.release(va.value)
                    aux.map(_.value).foreach(scope.release)
                  }
              }
            case _ =>
          }
        }
      case _ =>
    }

    mutable.toMap

  }

  private[lamp] def topologicalSort[D](root: Variable): Seq[Variable] = {
    type V = Variable
    var order = List.empty[V]
    var marks = Set.empty[AnyRef]
    var currentParents = Set.empty[AnyRef]

    def visit(n: V): Unit =
      if (marks.contains(n)) ()
      else {
        if (currentParents.contains(n)) {
          System.err.println(
            s"lamp autograd error: cycle detected with node (${n},${n.op}). This is an error in the framework. Operator's backward code might be ill defined."
          )
          ()
        } else {
          currentParents = currentParents + n
          val children =
            n.op.toList.flatMap(_.params.map(_.fold(_.variable, id => id)))
          children.foreach(visit)
          currentParents = currentParents - n
          marks = marks + n
          order = n :: order
        }
      }

    visit(root)

    order

  }

}

case class GraphMemoryAllocationReport(
    parameterTensorCount: Long,
    parameterTensorStorage: Long,
    constantTensorCount: Long,
    constantTensorStorage: Long,
    intermediateTensorCount: Long,
    intermediateTensorStorage: Long,
    pdTensorCount: Long,
    pdTensorStorage: Long
) {
  private def gb(l: Long) = f"${(l.toDouble * 1e-9)}%.4f"
  override def toString =
    s"#par=$parameterTensorCount(${gb(parameterTensorStorage)}GB);#const=$constantTensorCount(${gb(
      constantTensorStorage
    )}GB);#act=$intermediateTensorCount(${gb(intermediateTensorStorage)}GB;#PD=$pdTensorCount(${gb(pdTensorStorage)}GB)"
}
