package lamp.autograd
import org.saddle._
import org.saddle.ops.BinOps._
import org.saddle.linalg._
import java.{util => ju}
import aten.Tensor
import aten.ATen
import aten.TensorOptions
import TensorHelpers.{unbroadcast => ub}

case class Constant(const: Tensor) extends Op {
  val params = Nil
  val value = Variable(this, const, leaf = true)
  override def toString = s"CONST($const)"
}
case class Transpose(a: Variable) extends Op {
  val params = List(
    a.zipBackward { (p, out) =>
      val tmp = ATen.t(p)
      ATen.add_out(out, out, tmp, 1d)
      tmp.release()
    }
  )
  val value = Variable(this, ATen.t(a.value))
  override def toString = s"T(${a.stringify()})"
}

case class Add(a: Variable, b: Variable) extends Op {
  val params = List(
    a.zipBackward { (p, out) =>
      val p2 = ub(p, a.sizes)
      ATen.add_out(out, out, p2, 1d)
    },
    b.zipBackward { (p, out) =>
      val p2 = ub(p, b.sizes)
      ATen.add_out(out, out, p2, 1d)
    }
  )
  val value = Variable(this, ATen.add_0(a.value, b.value, 1d))

  override def toString = s"(${a.stringify()} + ${b.stringify()})"
}
case class Minus(a: Variable, b: Variable) extends Op {
  val params = List(
    a.zipBackward { (p, out) => ATen.add_out(out, out, ub(p, a.sizes), 1d) },
    b.zipBackward { (p, out) => ATen.add_out(out, out, ub(p, b.sizes), -1d) }
  )
  val value = Variable(this, ATen.add_0(a.value, b.value, -1d))

  override def toString = s"(${a.stringify()} - ${b.stringify()})"
}

case class Mult(a: Variable, b: Variable) extends Op {
  val params = List(
    a.zipBackward { (p, out) =>
      val tmp = ATen.mul_0(p, b.value)
      ATen.add_out(out, out, ub(tmp, a.sizes), 1d)
      tmp.release
    },
    b.zipBackward { (p, out) =>
      val tmp = ATen.mul_0(p, a.value)
      ATen.add_out(out, out, ub(tmp, b.sizes), 1d)
      tmp.release
    }
  )

  val value = Variable(this, ATen.mul_0(a.value, b.value))

  override def toString = s"(${a.stringify()} * ${b.stringify()})"
}
case class Div(a: Variable, b: Variable) extends Op {
  val params = List(
    a.zipBackward { (p, out) =>
      val tmp = ATen.div_0(p, b.value)
      ATen.add_out(out, out, ub(tmp, a.sizes), 1d)
      tmp.release
    },
    b.zipBackward { (p, out) =>
      // out += p * (a.value * b.value.map(x => -1d / (x * x)))
      val tmp = ATen.div_0(value.value, b.value)
      ATen.mul_out(tmp, tmp, p)
      ATen.add_out(out, out, ub(tmp, b.sizes), -1d)
      tmp.release()
    }
  )

  val value = Variable(this, ATen.div_0(a.value, b.value))

  override def toString = s"(${a.stringify()} / ${b.stringify()})"
}

case class Sum(a: Variable) extends Op {
  val params = List(a.zipBackward { (p, out) => ATen.add_out(out, out, p, 1d) })

  val value = Variable(this, ATen.sum_0(a.value))

  override def toString = s"SUM(${a.stringify()})"
}
case class ColSum(a: Variable) extends Op {
  val params = List(a.zipBackward { (p, out) => ATen.add_out(out, out, p, 1d) })

  val value = Variable(this, ATen.sum_1(a.value, Array(1), true))

  override def toString = s"COLSUM(${a.stringify()})"
}
case class RowSum(a: Variable) extends Op {
  val params = List(a.zipBackward { (p, out) => ATen.add_out(out, out, p, 1d) })

  val value = Variable(this, ATen.sum_1(a.value, Array(0), true))

  override def toString = s"RowSUM(${a.stringify()})"
}

// http://cs231n.stanford.edu/handouts/derivatives.pdf
case class MatMul(a: Variable, b: Variable) extends Op {
  val params =
    List(
      a.zipBackward { (p, out) =>
        val bt = ATen.t(b.value)
        ATen.addmm_out(out, out, p, bt, 1d, 1d)
        bt.release
      },
      b.zipBackward { (p, out) =>
        val at = ATen.t(a.value)
        ATen.addmm_out(out, out, at, p, 1d, 1d)
        at.release
      }
    )

  val value = Variable(this, ATen.mm(a.value, b.value))

  override def toString = s"(${a.stringify()} dot ${b.stringify()})"
}

case class Exp(a: Variable) extends Op {
  val params = List(
    a.zipBackward { (p, out) => ATen.addcmul_out(out, out, p, value.value, 1d) }
  )
  val value = Variable(this, ATen.exp(a.value))
  override def toString = s"EXP(${a.stringify()})"
}
case class Log(a: Variable) extends Op {
  val params = List(a.zipBackward { (p, out) =>
    val tmp = ATen.reciprocal(a.value)
    ATen.addcmul_out(out, out, p, tmp, 1d)
    tmp.release
  })
  val value = Variable(this, ATen.log(a.value))
  override def toString = s"LOG(${a.stringify()})"
}
case class Sin(a: Variable) extends Op {
  val params = List(a.zipBackward { (p, out) =>
    val tmp = ATen.cos(a.value)
    ATen.addcmul_out(out, out, p, tmp, 1d)
    tmp.release
  })
  val value = Variable(this, ATen.sin(a.value))
  override def toString = s"SIN(${a.stringify()})"
}
case class Cos(a: Variable) extends Op {
  val params = List(a.zipBackward { (p, out) =>
    val tmp = ATen.sin(a.value)
    ATen.addcmul_out(out, out, p, tmp, -1d)
    tmp.release
  })
  val value = Variable(this, ATen.cos(a.value))
  override def toString = s"COS(${a.stringify()})"
}
case class Tan(a: Variable) extends Op {
  val params = List(a.zipBackward { (p, out) =>
    val tmp1 = ATen.pow_0(value.value, 2d)
    val one =
      ATen.ones(Array(1L), tmp1.options)
    ATen.add_out(tmp1, tmp1, one, 1d)
    ATen.addcmul_out(out, out, p, tmp1, 1d)
    tmp1.release
    one.release
  })
  val value = Variable(this, ATen.tan(a.value))
  override def toString = s"TAN(${a.stringify()})"
}
case class ArcTan(a: Variable) extends Op {
  val params = List(a.zipBackward { (p, out) =>
    val tmp1 = ATen.pow_0(a.value, 2d)
    val one =
      ATen.ones(Array(1L), tmp1.options())
    ATen.add_out(tmp1, tmp1, one, 1d)
    ATen.reciprocal_(tmp1)
    ATen.addcmul_out(out, out, p, tmp1, 1d)
    tmp1.release
    one.release
  })
  val value = Variable(this, ATen.atan(a.value))
  override def toString = s"ATAN(${a.stringify()})"
}
case class PowConst(a: Variable, exponent: Double) extends Op {
  val params = List(a.zipBackward { (p, out) =>
    val tmp1 = ATen.pow_0(a.value, exponent - 1)
    ATen.addcmul_out(out, out, p, tmp1, exponent)
    tmp1.release
  })
  val value = Variable(this, ATen.pow_0(a.value, exponent))
  override def toString = s"POW(${a.stringify()},$exponent)"
}
case class Relu(a: Variable) extends Op {
  val params = List(
    a.zipBackward { (p, out) =>
      val pred = ATen.lt_0(a.value, 0d)
      val ones =
        ATen.ones(Array(1), a.value.options)
      val zeros =
        ATen.zeros(Array(1), a.value.options)
      val tmp = ATen.where_0(pred, zeros, ones)
      ATen.addcmul_out(out, out, p, tmp, 1d)
      tmp.release
      ones.release
      zeros.release
    }
  )
  val value = Variable(this, ATen.relu(a.value))
  override def toString = s"RELU(${a.stringify()})"
}

case class LogSoftMaxRowWise(a: Variable) extends Op {

  val params = List(
    a.zipBackward { (p, out) =>
      val tmp = ATen._log_softmax_backward_data(p, value.value, 1, a.value)
      ATen.add_out(out, out, tmp, 1d)
      tmp.release
    }
  )
  val value = Variable(this, ATen.log_softmax(a.value, 1))
  override def toString = s"LOGSOFTMAX(${a.stringify()})"

}
case class Gelu(a: Variable) extends Op {

  val params = List(
    a.zipBackward { (p, out) =>
      val tmp = ATen.gelu_backward(p, a.value)
      ATen.add_out(out, out, tmp, 1d)
      tmp.release
    }
  )
  val value = Variable(this, ATen.gelu(a.value))
  override def toString = s"GELU(${a.stringify()})"

}
case class Dropout(a: Variable, prob: Double, train: Boolean) extends Op {

  val params = List(
    a.zipBackward { (p, out) => ATen.addcmul_out(out, out, p, mask, 1d) }
  )
  val mask = {
    val ones = ATen.ones_like(a.value, a.options)
    ATen.dropout_(ones, prob, train)
    ones
  }
  val value = Variable(this, ATen.mul_0(a.value, mask))
  override def toString = s"DROPOUT(${a.stringify()})"

}

sealed trait Reduction {
  def asLong: Long
}
case object NoReduction extends Reduction {
  def asLong = 0L
}
case object Mean extends Reduction {
  def asLong = 1L
}
case object Sum extends Reduction {
  def asLong = 2L
}

case class NllLoss(
    input: Variable,
    target: Tensor,
    numClasses: Int,
    reduction: Reduction
) extends Op {
  assert(
    input.sizes.size == 2,
    "Nll Loss assumes 2D input (samples x classes). Higher dimensions not implemented."
  )
  assert(
    target.sizes.size == 1,
    "Target should be a 1D tensor with [0,C-1] integers, C number of classes."
  )
  val weights = ATen.ones(Array(numClasses), target.options.toDouble)
  val params = List(
    input.zipBackward { (p, out) =>
      val tmp =
        ATen.nll_loss_backward(
          p,
          input.value,
          target,
          weights,
          reduction.asLong,
          -100,
          total_weight
        )
      ATen.add_out(out, out, tmp, 1d)
      tmp.release
    }
  )
  val (value1, total_weight) =
    ATen.nll_loss_forward(input.value, target, weights, reduction.asLong, -100)

  val value =
    Variable(
      this,
      value1
    )
  override def toString = s"NLL(${input.stringify()})"

}

case class SquaredFrobeniusMatrixNorm(a: Variable) extends Op {
  val params = List(a.zipBackward { (p, out) =>
    ATen.addcmul_out(out, out, p, a.value, 2d)
  })
  val value =
    Variable(this, {
      val fr = ATen.frobenius_norm_0(a.value)
      ATen.pow_out_0(fr, fr, 2d)
      fr
    })
  override def toString = s"FROBENIUS(${a.stringify()})"
}

// // each row is a sample, batches are along the first dimension
// // https://arxiv.org/pdf/1502.03167.pdf
// // case class BatchNorm(a: Variable) extends Op
