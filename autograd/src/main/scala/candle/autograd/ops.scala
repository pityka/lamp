package candle.autograd
import org.saddle._
import org.saddle.ops.BinOps._
import org.saddle.linalg._
import java.{util => ju}
import aten.Tensor
import aten.ATen

// case class Relu(a: Variable) extends ElementwiseOp {

//   def op(d: Double) = if (d < 0d) 0d else d
//   def diff(d: Double) = if (d < 0d) 0d else 1d

//   override def toString = s"RELU(${a.stringify()})"
// }

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
// case class Div(a: Variable, b: Variable) extends Op {
//   val params = List(
//     a.zipBackward((p, out) => {
//       out += p * b.value.map(x => 1d / x)
//     }),
//     b.zipBackward((p, out) =>
//       out += p * (a.value * b.value.map(x => -1d / (x * x)))
//     )
//   )

//   val value = Variable(this, a.value / b.value)

//   override def toString = s"(${a.stringify()} / ${b.stringify()})"
// }

case class Sum(a: Variable) extends Op {
  val params = List(a.zipBackward { (p, out) => ATen.add_out(out, out, p, 1d) })

  val value = Variable(this, ATen.sum_0(a.value))

  override def toString = s"SUM(${a.stringify()})"
}
// case class ColSum(a: Variable) extends Op {
//   val params = List(a.zipBackward((p, out) => {
//     out += (p * mat.ones(a.value.numRows, a.value.numCols))
//   }))

//   val value = Variable(this, Mat(a.value.colSums).T)

//   override def toString = s"COLSUM(${a.stringify()})"
// }
// case class RowSum(a: Variable) extends Op {
//   val params = List(a.zipBackward((p, out) => {
//     out += (p * mat.ones(a.value.numRows, a.value.numCols))
//   }))

//   val value = Variable(this, Mat(a.value.rowSums))

//   override def toString = s"ROWSUM(${a.stringify()})"
// }

// // http://cs231n.stanford.edu/handouts/derivatives.pdf
// case class MatMul(a: Variable, b: Variable) extends Op {
//   val params =
//     List(
//       a.zipBackward((p, out) => out += p mmt b.value),
//       b.zipBackward((p, out) => out += a.value tmm p)
//     )

//   val value = Variable(this, a.value mm b.value)

//   override def toString = s"(${a.stringify()} dot ${b.stringify()})"
// }

// trait ElementwiseOp extends Op {
//   def a: Variable
//   def op(d: Double): Double
//   def diff(d: Double): Double

//   val params = List(
//     a.zipBackward((p, out) => out += p * a.value.map(diff))
//   )
//   val value = Variable(this, a.value.map(op))

// }

// case class Exp(a: Variable) extends ElementwiseOp {
//   def op(d: Double) = math.exp(d)
//   def diff(d: Double) = math.exp(d)
//   override def toString = s"EXP(${a.stringify()})"
// }
// case class Log(a: Variable) extends ElementwiseOp {
//   def op(d: Double) = math.log(d)
//   def diff(d: Double) = 1d / d

//   override def toString = s"LOG(${a.stringify()})"
// }
// case class Sin(a: Variable) extends ElementwiseOp {
//   def op(d: Double) = math.sin(d)
//   def diff(d: Double) = math.cos(d)

//   override def toString = s"SIN(${a.stringify()})"
// }
// case class Cos(a: Variable) extends ElementwiseOp {
//   def op(d: Double) = math.cos(d)
//   def diff(d: Double) = -math.sin(d)

//   override def toString = s"COS(${a.stringify()})"
// }
// case class Tan(a: Variable) extends ElementwiseOp {
//   def op(d: Double) = math.tan(d)
//   def diff(d: Double) = 1 + math.pow(math.tan(d), 2d)

//   override def toString = s"COS(${a.stringify()})"
// }
// case class ArcTan(a: Variable) extends ElementwiseOp {
//   def op(d: Double) = math.atan(d)
//   def diff(d: Double) = 1d / (1d + d * d)

//   override def toString = s"COS(${a.stringify()})"
// }

// case class PowConst(a: Variable, param: Double) extends ElementwiseOp {
//   def op(d: Double) = math.pow(d, param)
//   def diff(d: Double) = param * math.pow(d, param - 1d)

//   override def toString = s"POW(${a.stringify()},$param)"
// }

// trait RowWiseOp extends Op {
//   def a: Variable
//   def op(d: Vec[Double]): Vec[Double]
//   def diff(rowIdx: Int): Mat[Double]

//   val params = List(
//     a.zipBackward { (p, out) =>
//       out += p.mapRows { (prow, idx) =>
//         val d = diff(idx)
//         (Mat(prow) tmm d).row(0)
//       }
//     }
//   )
//   val value = Variable(this, a.value.mapRows { (row, _) => op(row) })

// }

// case class LogSoftMaxRowWise(a: Variable) extends RowWiseOp {

//   def diff(rowIdx: Int) = {
//     mat.ident(a.value.numCols) + value.value
//       .row(Array(rowIdx))
//       .map(x => -math.exp(x))
//   }

//   private def logSumExp(row: Vec[Double]) = {
//     val max = row.max2
//     math.log(row.map(e => math.exp(e - max)).sum2) + max
//   }
//   def op(row: Vec[Double]) = {
//     val l = logSumExp(row)
//     row.map(x => x - l)
//   }
//   override def toString = s"LOGSOFTMAX(${a.stringify()})"
// }

// // computes -(reference dot logQuery) for each row
// case class CrossEntropyRowWiseLog(logQuery: Variable, reference: Variable)
//     extends Op {

//   override val params = List(
//     reference.zipBackward { (p, out) => out -= p * logQuery.value },
//     logQuery.zipBackward { (p, out) => out -= p * reference.value }
//   )

//   val value =
//     Variable(
//       this,
//       Mat(
//         reference.value.rows
//           .zip(logQuery.value.rows)
//           .map {
//             case (rowa, rowb) =>
//               (rowa vv rowb) * -1
//           }
//           .toVec
//       )
//     )
//   override def toString =
//     s"CROSSENTROPY(${reference.stringify()} , ${logQuery.stringify()})"
// }

// case class SquaredFrobeniusMatrixNorm(a: Variable) extends Op {
//   val params = List(
//     a.zipBackward { (p, out) => out += p * (a.value * 2) }
//   )
//   val value =
//     Variable(this, Mat(Vec(a.value.map(x => x * x).toVec.sum2)))
//   override def toString = s"FROBENIUS(${a.stringify()})"
// }

// // each row is a sample, batches are along the first dimension
// // https://arxiv.org/pdf/1502.03167.pdf
// // case class BatchNorm(a: Variable) extends Op
