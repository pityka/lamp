package lamp.autograd
import org.saddle._
import org.saddle.ops.BinOps._
import org.saddle.linalg._
import java.{util => ju}
import aten.Tensor
import aten.ATen
import aten.TensorOptions

case class Constant(const: Tensor) extends Op {
  val params = Nil
  val value = Variable(this, const)
  override def toString = s"$const"
}

case class Add(a: Variable, b: Variable) extends Op {
  val params = List(
    a.zipBackward { (p, out) => ATen.add_out(out, out, p, 1d) },
    b.zipBackward { (p, out) => ATen.add_out(out, out, p, 1d) }
  )
  val value = Variable(this, ATen.add_0(a.value, b.value, 1d))

  override def toString = s"(${a.stringify()} + ${b.stringify()})"
}
case class Minus(a: Variable, b: Variable) extends Op {
  val params = List(
    a.zipBackward { (p, out) => ATen.add_out(out, out, p, 1d) },
    b.zipBackward { (p, out) => ATen.add_out(out, out, p, -1d) }
  )
  val value = Variable(this, ATen.add_0(a.value, b.value, -1d))

  override def toString = s"(${a.stringify()} - ${b.stringify()})"
}

case class Mult(a: Variable, b: Variable) extends Op {
  val params = List(
    a.zipBackward { (p, out) => ATen.addcmul_out(out, out, p, b.value, 1d) },
    b.zipBackward { (p, out) => ATen.addcmul_out(out, out, p, a.value, 1d) }
  )

  val value = Variable(this, ATen.mul_0(a.value, b.value))

  override def toString = s"(${a.stringify()} * ${b.stringify()})"
}
case class Div(a: Variable, b: Variable) extends Op {
  val params = List(
    a.zipBackward { (p, out) => ATen.addcdiv_out(out, out, p, b.value, 1d) },
    b.zipBackward { (p, out) =>
      // out += p * (a.value * b.value.map(x => -1d / (x * x)))
      val tmp = ATen.div_0(value.value, b.value)
      ATen.addcmul_out(out, out, p, tmp, -1d)
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
      ATen.ones(Array(1L), TensorOptions.fromScalarType(tmp1.scalarType))
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
      ATen.ones(Array(1L), TensorOptions.fromScalarType(tmp1.scalarType))
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
        ATen.ones(Array(1), TensorOptions.fromScalarType(a.value.scalarType()))
      val zeros =
        ATen.zeros(Array(1), TensorOptions.fromScalarType(a.value.scalarType()))
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
