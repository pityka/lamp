package lamp.autograd
import aten.Tensor
import aten.ATen
import lamp.TensorHelpers.{unbroadcast => ub}
import lamp.util.syntax
import lamp.FloatingPointPrecision
import lamp.DoublePrecision
import lamp.SinglePrecision

import lamp.Scope
import lamp.STen
import lamp.STenOptions

case class Transpose(scope: Scope, a: Variable, dim1: Int = 0, dim2: Int = 1)
    extends Op {

  val params = List(
    a.zipBackward { (p, out) =>
      Scope.root { implicit scope => out += p.transpose(dim1, dim2) }

    }
  )
  val value = Variable(
    this,
    a.value.transpose(dim1, dim2)(scope)
  )(scope)
}

case class View(scope: Scope, a: Variable, shape: Array[Long]) extends Op {

  val params = List(
    a.zipBackward { (p, out) =>
      Scope.root { implicit scope => out += p.view(out.shape: _*) }

    }
  )
  val value = Variable(this, a.value.view(shape.toSeq: _*)(scope))(scope)
}
case class Reshape(scope: Scope, a: Variable, shape: Array[Long]) extends Op {

  val params = List(
    a.zipBackward { (p, out) =>
      Scope.root { implicit scope => out += p.reshape(out.shape: _*) }

    }
  )
  val value = Variable(this, a.value.reshape(shape.toSeq: _*)(scope))(scope)
}

case class Concatenate(scope: Scope, a: Seq[Variable], dim: Long) extends Op {
  val ashapes = a.map(_.shape(dim.toInt))
  val boundaries =
    ashapes.scanLeft(0L)(_ + _).sliding(2).toList.map(g => g(0) -> g(1))
  val params = a.zip(boundaries).toList.map { case (a, (from, to)) =>
    a.zipBackward { (p, out) =>
      Scope.root { implicit scope => out += p.slice(dim, from, to, 1L) }
    }
  }
  val value =
    Variable(this, STen.cat(a.map(_.value), dim)(scope))(scope)
}

case class Stack(scope: Scope, a: Seq[Variable], dim: Long) extends Op {
  val params = a.zipWithIndex.toList.map { case (a, idx) =>
    a.zipBackward { (p, out) =>
      Scope.root { implicit scope => out += p.select(dim, idx) }
    }
  }
  val value =
    Variable(this, STen.stack(a.map(_.value), dim)(scope))(scope)
}

case class Select(scope: Scope, a: Variable, dim: Long, index: Long)
    extends Op {

  val params = List(
    a.zipBackward { (p, out) =>
      Scope.root { implicit scope =>
        val tmp = STen.zeros(out.sizes, a.options)
        val scalar = STen.scalarLong(index, a.options)
        val pshape = p.shape
        val reshape =
          pshape.take(dim.toInt) ::: (1L :: pshape.drop(dim.toInt))
        val p2 = p.view(reshape: _*)
        val tmp2 = tmp.indexAdd(dim, scalar, p2)
        out += tmp2

      }

    }
  )
  val value =
    Variable(this, a.value.select(dim, index)(scope))(scope)
}
case class EqWhere(scope: Scope, a: Variable, b: Long) extends Op {

  val params = List()
  val value = Variable(
    this, {
      implicit val sc = scope
      Scope { implicit sc =>
        val r = a.value.equ(b)
        r
      }
    }
  )(scope)
}
case class MaskSelect(scope: Scope, input: Variable, mask: Variable)
    extends Op {

  val params = List(
    input.zipBackward { (p, out) =>
      Scope.root { implicit scope =>
        val tmp = STen.zerosLike(out)
        out += tmp.maskedScatter(mask.value, p)
      }

    }
  )
  val value =
    Variable(this, input.value.maskedSelect(mask.value)(scope))(scope)
}
case class MaskFill(scope: Scope, input: Variable, mask: Variable, fill: Double)
    extends Op {

  val params = List(
    input.zipBackward { (p, out) =>
      Scope.root { implicit scope => out += p.maskedFill(mask.value, 0d) }

    }
  )
  val value =
    Variable(this, input.value.maskFill(mask.value, fill)(scope))(scope)
}
case class IndexFill(
    scope: Scope,
    input: Variable,
    dim: Long,
    index: Variable,
    fill: Double
) extends Op {

  val params = List(
    input.zipBackward { (p, out) =>
      Scope.root { implicit scope => out += p.indexFill(dim, index.value, 0d) }

    }
  )
  val value =
    Variable(this, input.value.indexFill(dim, index.value, fill)(scope))(
      scope
    )
}
case class IndexSelect(
    scope: Scope,
    input: Variable,
    dim: Long,
    index: Variable
) extends Op {

  val params = List(
    input.zipBackward { (p, out) =>
      Scope.root { implicit scope =>
        val tmp = out.indexAdd(dim, index.value, p)
        out += tmp
      }

    }
  )
  val value =
    Variable(this, input.value.indexSelect(dim, index.value)(scope))(scope)
}
case class Where(
    scope: Scope,
    condition: STen,
    trueBranch: Variable,
    falseBranch: Variable
) extends Op {

  val params = List(
    trueBranch.zipBackward { (p, out) =>
      Scope.root { implicit scope =>
        val ones = STen.onesLike(trueBranch.value)
        val zeros = STen.zerosLike(falseBranch.value)
        val tmp = STen.where(condition, ones, zeros)
        out.addcmulSelf(p, tmp, 1.0)
      }
    },
    falseBranch.zipBackward { (p, out) =>
      Scope.root { implicit scope =>
        val zeros = STen.zerosLike(trueBranch.value)
        val ones = STen.onesLike(falseBranch.value)
        val tmp = STen.where(condition, zeros, ones)
        out.addcmulSelf(p, tmp, 1.0)
      }

    }
  )
  val value =
    Variable(
      this,
      STen.where(condition, trueBranch.value, falseBranch.value)(scope)
    )(scope)
}
case class ArgMax(scope: Scope, a: Variable, dim: Long, keepDim: Boolean)
    extends Op {

  val params = List(
    a.zipBackward { (_, _) =>
      throw new RuntimeException("Argmax is not differentiable")
    }
  )
  val value =
    Variable(this, a.value.argmax(dim, keepDim)(scope))(scope)
}

case class Assign(scope: Scope, abandon: Variable, keep: Variable) extends Op {

  val params = List(
    abandon.zipBackward { (_, _) => () },
    keep.zipBackward { (p, out) => out += p }
  )
  val value = Variable(this, keep.value)(scope)
}
case class OneHot(scope: Scope, a: Variable, numClasses: Int) extends Op {

  val params = List(
    a.zipBackward { (_, _) =>
      throw new RuntimeException("OneHot is not differentiable")
    }
  )
  val value =
    Variable(this, a.value.oneHot(numClasses)(scope))(scope)
}
case class CastToPrecision(
    scope: Scope,
    a: Variable,
    precision: FloatingPointPrecision
) extends Op {

  val params = List(
    a.zipBackward { (_, _) =>
      throw new RuntimeException("Cast to Precision is not differentiable")
    }
  )
  val value = Variable(
    this, {
      precision match {
        case DoublePrecision => a.value.castToDouble(scope)
        case SinglePrecision => a.value.castToFloat(scope)
      }
    }
  )(scope)
}
case class SparseFromValueAndIndex(
    scope: Scope,
    values: Variable,
    indices: STen,
    dim: Seq[Long]
) extends Op {

  val params = List(
    values.zipBackward { case (p, out) =>
      Scope.root { implicit scope =>
        out += p.values
      }
    }
  )
  val value = Variable(
    this,
    STen.sparse_coo(
      indices = indices,
      values = values.value,
      dim = dim,
      values.value.device.to(
        STenOptions
          .fromScalarType(values.value.scalarTypeByte)(scope)
      )(scope)
    )(scope)
  )(scope)
}
case class ToDense(
    scope: Scope,
    sparse: Variable
) extends Op {

  val params = List(
    sparse.zipBackward { case (p, out) =>
      Scope.root { implicit scope =>
        out += STen.to_dense_backward(p, sparse.value)
      }
    }
  )
  val value = Variable(
    this,
    sparse.value.toDense(scope)
  )(scope)
}
case class Diag(
    scope: Scope,
    a: Variable,
    diagonal: Long
) extends Op {

  val params = List(
    a.zipBackward { case (p, out) =>
      Scope.root { implicit scope =>
        out += p.diag(diagonal)
      }
    }
  )
  val value = Variable(
    this,
    a.value.diag(diagonal)(scope)
  )(scope)
}
case class Inv(
    scope: Scope,
    a: Variable
) extends Op {
  val params = List(
    a.zipBackward { case (p, out) =>
      Scope.root { implicit scope =>
        if (value.shape.size == 3) {
          out -= value.value.bmm(p).bmm(value.value).transpose(1, 2)
        } else {
          out -= value.value.mm(p).mm(value.value).t
        }
      }
    }
  )
  val value = Variable(
    this,
    a.value.inv(scope)
  )(scope)
}
case class PInv(
    scope: Scope,
    a: Variable,
    rcond: Double
) extends Op {
  val params = List(
    a.zipBackward { case (p, out) =>
      Scope.root { implicit scope =>
        // https://math.stackexchange.com/questions/2179160/derivative-of-pseudoinverse-with-respect-to-original-matrix
        if (value.shape.size == 3) {
          val v = value.value
          val vt = v.transpose(1, 2)
          val pt = p.transpose(1, 2)
          val i = STen.eye(a.shape(1).toInt, v.options).unsqueeze(0)

          out += v.bmm(vt).bmm(pt).bmm(i - a.value.bmm(v)) + (i - v
            .bmm(a.value))
            .bmm(pt)
            .bmm(vt.bmm(v)) - v.bmm(p.bmm(v))
        } else {
          val v = value.value
          val vt = v.t
          val pt = p.t
          val i = STen.eye(a.shape(0).toInt, v.options)

          out += v.mm(vt).mm(pt).mm(i - a.value.mm(v)) + (i - v
            .mm(a.value))
            .mm(pt)
            .mm(vt.mm(v)) - v.mm(p.mm(v))
        }
      }
    }
  )
  val value = Variable(
    this,
    a.value.pinverse(rcond)(scope)
  )(scope)
}

case class ScatterAdd(
    scope: Scope,
    src: Variable,
    index: Variable,
    dim: Int,
    maxIndex: Long
) extends Op {

  assert(src.shape(dim) == index.shape(dim))
  val params = List(
    src.zipBackward { (p, out) =>
      Scope.root { implicit scope => out += p.gather(dim, index.value) }

    }
  )
  val value = {
    val shape = src.sizes.toArray
    shape(dim) = maxIndex
    implicit val sc = scope
    val result = Scope { implicit sc =>
      val zeros = STen.zeros(shape.toList, src.options)
      zeros.scatterAdd(dim, index.value, src.value)
    }
    Variable(this, result)(scope)
  }
}
case class IndexAdd(
    scope: Scope,
    src: Variable,
    index: Variable,
    dim: Int,
    maxIndex: Long
) extends Op {

  val params = List(
    src.zipBackward { (p, out) =>
      Scope.root { implicit scope => out += p.indexSelect(dim, index.value) }

    }
  )
  val value = {
    val shape = src.sizes.toArray
    shape(dim) = maxIndex
    implicit val sc = scope
    val result = Scope { implicit sc =>
      val zeros = STen.zeros(shape.toList, src.options)
      zeros.indexAdd(dim, index.value, src.value)
    }

    Variable(this, result)(scope)
  }
}
case class IndexAddToTarget(
    scope: Scope,
    target: Variable,
    src: Variable,
    index: Variable,
    dim: Int
) extends Op {

  val params = List(
    src.zipBackward { (p, out) =>
      Scope.root { implicit scope => out += p.indexSelect(dim, index.value) }
    },
    target.zipBackward { (p, out) =>
      out += p
    }
  )
  val value =
    Variable(this, target.value.indexAdd(dim, index.value, src.value)(scope))(
      scope
    )

}
case class RepeatInterleave(
    scope: Scope,
    self: Variable,
    repeats: Variable,
    dim: Int
) extends Op {

  val params = List(
    self.zipBackward { (p, out) =>
      Scope.root { implicit scope =>
        val plainIndices =
          STen.arange(0, self.shape(0).toDouble, 1d, self.options.toLong)
        val repeatedIndices = plainIndices.repeatInterleave(repeats.value, 0)
        val zeros = STen.zerosLike(out)
        val added = zeros.indexAdd(0, repeatedIndices, p)
        out += added
      }

    }
  )

  val value = Variable(
    this,
    self.value.repeatInterleave(repeats.value, dim)(scope)
  )(scope)
}

case class Add(scope: Scope, a: Variable, b: Variable) extends Op {

  val params = List(
    a.zipBackward { (p, out) =>
      Scope.root { implicit scope => out += p.unbroadcast(a.sizes) }
    },
    b.zipBackward { (p, out) =>
      Scope.root { implicit scope => out += p.unbroadcast(b.sizes) }

    }
  )
  val value =
    Variable(this, a.value.+(b.value)(scope))(scope)

}
case class ConstAdd(scope: Scope, a: Variable, b: Double) extends Op {

  val params = List(
    a.zipBackward { (p, out) =>
      Scope.root { implicit scope => out += p.unbroadcast(a.sizes) }

    }
  )
  val value = Variable(this, a.value.+(b)(scope))(scope)

}
case class Minus(scope: Scope, a: Variable, b: Variable) extends Op {

  val params = List(
    a.zipBackward { (p, out) =>
      Scope.root { implicit scope => out += p.unbroadcast(a.sizes) }

    },
    b.zipBackward { (p, out) =>
      Scope.root { implicit scope => out -= p.unbroadcast(b.sizes) }

    }
  )
  val value =
    Variable(this, a.value.-(b.value)(scope))(scope)

}
case class ConstMult(scope: Scope, a: Variable, b: Double) extends Op {

  val params = List(
    a.zipBackward { (p, out) =>
      Scope.root { implicit scope => out += (p * b).unbroadcast(a.sizes) }

    }
  )

  val value = Variable(this, a.value.*(b)(scope))(scope)

}
case class Mult(scope: Scope, a: Variable, b: Variable) extends Op {

  val params = List(
    a.zipBackward { (p, out) =>
      Scope.root { implicit scope => out += (p * b.value).unbroadcast(a.sizes) }

    },
    b.zipBackward { (p, out) =>
      Scope.root { implicit scope => out += (p * a.value).unbroadcast(b.sizes) }

    }
  )

  val value = Variable(this, a.value.*(b.value)(scope))(scope)

}
case class Cross(scope: Scope, a: Variable, b: Variable, dim: Int) extends Op {

  val params = List(
    a.zipBackward { (p, out) =>
      Scope.root { implicit scope =>
        out -= p * (STen.ones(a.shape, p.options).cross(b.value, dim))

      }

    },
    b.zipBackward { (p, out) =>
      Scope.root { implicit scope =>
        out += p * (STen.ones(b.shape, p.options).cross(a.value, dim))
      }

    }
  )

  val value = Variable(this, a.value.cross(b.value, dim)(scope))(scope)

}
case class Div(scope: Scope, a: Variable, b: Variable) extends Op {

  val params = List(
    a.zipBackward { (p, out) =>
      Scope.root { implicit scope => out += (p / b.value).unbroadcast(a.sizes) }

    },
    b.zipBackward { (p, out) =>
      // out += p * (a.value * b.value.map(x => -1d / (x * x)))
      Scope.root { implicit scope =>
        val tmp = value.value / b.value
        tmp *= p
        val t2 = tmp.unbroadcast(b.sizes)
        out -= t2
      }
    }
  )

  val value = Variable(this, a.value./(b.value)(scope))(scope)
}

case class Sum(scope: Scope, a: Variable, dim: List[Int], keepDim: Boolean)
    extends Op {

  val params = List(a.zipBackward { (p, out) => out += p })

  val value = Variable(this, a.value.sum(dim, keepDim)(scope))(scope)

}

case class ExpandAs(scope: Scope, a: Variable, as: STen) extends Op {

  val params = List(a.zipBackward { (p, out) =>
    Scope.root { implicit scope => out += p.unbroadcast(a.shape) }
  })
  val value =
    Variable(this, a.value.expandAs(as)(scope))(scope)
}

// http://cs231n.stanford.edu/handouts/derivatives.pdf
case class MatMul(scope: Scope, a: Variable, b: Variable) extends Op {

  val params =
    List(
      a.zipBackward { (p, out) =>
        Tensor
          .addmm_out_transposed2(
            out.value,
            out.value,
            p.value,
            b.value.value,
            1d,
            1d
          )
      },
      b.zipBackward { (p, out) =>
        Tensor
          .addmm_out_transposed1(
            out.value,
            out.value,
            a.value.value,
            p.value,
            1d,
            1d
          )
      }
    )

  val value = Variable(this, a.value.mm(b.value)(scope))(scope)

}
case class BatchedMatMul(scope: Scope, a: Variable, b: Variable) extends Op {

  val params =
    List(
      a.zipBackward { (p, out) =>
        Tensor.baddbmm_out_transposed2(
          out.value,
          out.value,
          p.value,
          b.value.value,
          1d,
          1d
        )
      },
      b.zipBackward { (p, out) =>
        Tensor.baddbmm_out_transposed1(
          out.value,
          out.value,
          a.value.value,
          p.value,
          1d,
          1d
        )
      }
    )

  val value = Variable(this, a.value.bmm(b.value)(scope))(scope)

}
case class EuclideanDistance(scope: Scope, a: Variable, b: Variable, dim: Int)
    extends Op {

  val params =
    List(
      a.zipBackward { (p, out) =>
        Scope.root { implicit scope =>
          val tmp = diff / norm
          out.addcmulSelf(p, tmp, 1d)
        }

      },
      b.zipBackward { (p, out) =>
        Scope.root { implicit scope =>
          val tmp = diff / norm
          out.addcmulSelf(p, tmp, -1d)
        }

      }
    )
  val diff = a.value.-(b.value)(scope)

  val norm =
    diff.norm2(List(dim), true)(scope)

  val value = Variable(this, norm)(scope)

}

case class Exp(scope: Scope, a: Variable) extends Op {

  val params = List(a.zipBackward { (p, out) =>
    out.addcmulSelf(p, value.value, 1d)
  })
  val value = Variable(this, a.value.exp(scope))(scope)
}
case class CappedShiftedNegativeExponential(
    scope: Scope,
    a: Variable,
    shift: Double
) extends Op {

  val params = List(
    a.zipBackward { (p, out) =>
      Scope.root { implicit scope =>
        val zeros = STen.zeros(List(1), aOpt)
        val nonzeros = STen.owned(ATen.mul_1(result, -1d))
        val tmp = STen.where(pred, zeros, nonzeros)
        out.addcmulSelf(p, tmp, 1d)
      }
    }
  )
  val aOpt = a.options(scope)
  val pred = scope.apply(ATen.le_0(a.value.value, shift))
  val ones =
    scope.apply(ATen.ones(Array(1), aOpt.value))
  val scalar = scope.apply(ATen.scalar_tensor(shift, aOpt.value))
  val above = scope.apply(ATen.sub_0(scalar, a.value.value, 1d))
  ATen.exp_(above)
  val result = ATen.where_0(pred, ones, above)
  val value = Variable(this, STen.owned(result)(scope))(scope)
}
case class LogDet(scope: Scope, a: Variable) extends Op {

  val params = List(a.zipBackward { (p, out) =>
    Scope.root { implicit scope =>
      val tmp = a.value.inv.t
      out.addcmulSelf(p, tmp, 1d)
    }
  })
  val value = Variable(this, a.value.det(scope).log(scope))(scope)
}
case class Log(scope: Scope, a: Variable) extends Op {

  val params = List(a.zipBackward { (p, out) =>
    Scope.root { implicit scope =>
      val tmp = a.value.reciprocal
      out.addcmulSelf(p, tmp, 1d)
    }
  })
  val value = Variable(this, a.value.log(scope))(scope)
}
case class Log1p(scope: Scope, a: Variable) extends Op {

  val params = List(a.zipBackward { (p, out) =>
    Scope.root { implicit scope =>
      val tmp = a.value + 1d
      tmp.reciprocal_()
      out.addcmulSelf(p, tmp, 1d)
    }

  })
  val value = Variable(this, a.value.log1p(scope))(scope)
}
case class Sin(scope: Scope, a: Variable) extends Op {

  val params = List(a.zipBackward { (p, out) =>
    Scope.root { implicit scope =>
      val tmp = a.value.cos
      out.addcmulSelf(p, tmp, 1d)
    }

  })
  val value = Variable(this, a.value.sin(scope))(scope)
}
case class Cos(scope: Scope, a: Variable) extends Op {

  val params = List(a.zipBackward { (p, out) =>
    Scope.root { implicit scope =>
      val tmp = a.value.sin
      out.addcmulSelf(p, tmp, -1d)
    }

  })
  val value = Variable(this, a.value.cos(scope))(scope)
}
case class Tan(scope: Scope, a: Variable) extends Op {

  val params = List(a.zipBackward { (p, out) =>
    Scope.root { implicit scope =>
      val tmp1 = value.value.pow(2d)
      val one = STen.ones(List(1), a.options)
      tmp1 += one
      out.addcmulSelf(p, tmp1, 1d)
    }

  })
  val value = Variable(this, a.value.tan(scope))(scope)
}
case class Tanh(scope: Scope, a: Variable) extends Op {

  val params = List(a.zipBackward { (p, out) =>
    Scope.root { implicit scope =>
      val tmp1 = STen.tanh_backward(p, value.value)
      out += tmp1
    }
  })
  val value = Variable(this, a.value.tanh(scope))(scope)
}
case class ArcTan(scope: Scope, a: Variable) extends Op {

  val params = List(a.zipBackward { (p, out) =>
    Scope.root { implicit scope =>
      val tmp1 = a.value.pow(2d)
      val one = STen.ones(List(1L), a.options)
      tmp1 += one
      tmp1.reciprocal_()
      out.addcmulSelf(p, tmp1, 1d)

    }

  })
  val value = Variable(this, a.value.atan(scope))(scope)
}
case class PowConst(scope: Scope, a: Variable, exponent: Double) extends Op {

  val params = List(a.zipBackward { (p, out) =>
    Scope.root { implicit scope =>
      val tmp1 = a.value.pow(exponent - 1)
      out.addcmulSelf(p, tmp1, exponent)
    }

  })
  val value = Variable(this, a.value.pow(exponent)(scope))(scope)
}
case class Pow(scope: Scope, a: Variable, exponent: Variable) extends Op {

  val params = List(
    a.zipBackward { (p, out) =>
      Scope.root { implicit scope =>
        val exp = exponent.toMat.raw(0)
        val tmp1 = a.value.pow(exp - 1)
        out.addcmulSelf(p, tmp1, exp)
      }

    },
    exponent.zipBackward { (p, out) =>
      Scope.root { implicit scope =>
        val exp = exponent.toMat.raw(0)
        val tmp1 = a.value.pow(exp)
        val tmp2 = a.value.log
        val tmp3 = tmp1 * tmp2
        val p2 = p.unbroadcast(
          List(if (out.sizes.isEmpty) 1 else out.sizes.toList.head, 1)
        )
        val tmp4 = tmp3.sum
        out.addcmulSelf(p2, tmp4, 1d)
      }
    }
  )
  val value =
    Variable(this, a.value.pow(exponent.value)(scope))(scope)
}
case class Relu(scope: Scope, a: Variable) extends Op {

  val params = List(
    a.zipBackward { (p, out) =>
      Scope.root { implicit scope =>
        val pred = a.value.lt(0d)
        val ones =
          STen.ones(List(1), aOpt)
        val zeros =
          STen.zeros(List(1), aOpt)
        val tmp = STen.where(pred, zeros, ones)
        out.addcmulSelf(p, tmp, 1d)
      }
    }
  )
  val aOpt = a.options(scope)
  val value = Variable(this, a.value.relu(scope))(scope)
}
case class LeakyRelu(scope: Scope, a: Variable, slope: Double) extends Op {

  val params = List(
    a.zipBackward { (p, out) =>
      Scope.root { implicit scope =>
        val pred = a.value.lt(0d)
        val ones =
          STen.ones(List(1), aOpt)
        val s =
          STen.zeros(List(1), aOpt) + slope
        val tmp = STen.where(pred, s, ones)
        out.addcmulSelf(p, tmp, 1d)
      }
    }
  )
  val aOpt = a.options(scope)
  val value = Variable(this, a.value.leakyRelu(slope)(scope))(scope)
}

case class LogSoftMax(scope: Scope, a: Variable, dim: Int) extends Op {

  val params = List(
    a.zipBackward { (p, out) =>
      Scope.root { implicit scope =>
        val tmp = STen.owned(
          ATen._log_softmax_backward_data(
            p.value,
            value.value.value,
            dim,
            a.value.value
          )
        )
        out += tmp

      }
    }
  )
  val value = Variable(this, a.value.logSoftMax(dim)(scope))(scope)

}
case class Gelu(scope: Scope, a: Variable) extends Op {

  val params = List(
    a.zipBackward { (p, out) =>
      Scope.root { implicit scope =>
        val tmp = STen.owned(ATen.gelu_backward(p.value, a.value.value))
        out += tmp
      }
    }
  )
  val value = Variable(this, a.value.gelu(scope))(scope)

}
case class Softplus(scope: Scope, a: Variable, beta: Double, threshold: Double)
    extends Op {

  val params = List(
    a.zipBackward { (p, out) =>
      Scope.root { implicit scope =>
        val tmp =
          STen.softplus_backward(p, a.value, beta, threshold, value.value)
        out += tmp
      }
    }
  )
  val value = Variable(this, a.value.softplus(beta, threshold)(scope))(scope)

}
case class Sigmoid(scope: Scope, a: Variable) extends Op {

  val params = List(
    a.zipBackward { (p, out) =>
      Scope.root { implicit scope =>
        val tmp = STen.owned(ATen.sigmoid_backward(p.value, value.value.value))
        out += tmp
      }

    }
  )
  val value = Variable(this, a.value.sigmoid(scope))(scope)

}

case class Mean(scope: Scope, a: Variable, dim: List[Int], keepDim: Boolean)
    extends Op {

  val params = List(
    a.zipBackward { (p, out) =>
      STen.addOut(out, out, p, 1d / dim.map(l => a.sizes.apply(l.toInt)).sum)

    }
  )
  val value =
    Variable(
      this,
      a.value.mean(dim, keepDim)(scope)
    )(scope)

}
case class Variance(scope: Scope, a: Variable, dim: List[Int]) extends Op {

  val params = List(
    a.zipBackward { (p, out) =>
      Scope.root { implicit scope =>
        val s = a.value - m
        out.addcmulSelf(
          p,
          s,
          2d / (dim.map(l => a.sizes.apply(l.toInt)).sum - 1)
        )

      }

    }
  )
  val (v, m) = a.value.varAndMean(dim, true, true)(scope)
  val value =
    Variable(
      this,
      v
    )(scope)

}
case class Dropout(scope: Scope, a: Variable, prob: Double, train: Boolean)
    extends Op {

  val params = List(
    a.zipBackward { (p, out) => out.addcmulSelf(p, mask, 1d) }
  )
  val mask = {
    val ones = STen.onesLike(a.value)(scope)
    ones.dropout_(prob, train)
    ones
  }
  val value =
    Variable(this, a.value.*(mask)(scope))(scope)

}

// https://arxiv.org/pdf/1602.07868.pdf
case class WeightNorm(scope: Scope, v: Variable, g: Variable, dim: Long)
    extends Op {

  assert(v.sizes.size == 2, "WeightNorm: v should have 2 dimensions")
  assert(
    g.sizes.toList == List(1, v.sizes(1)),
    "WeightNorm: g should have dimensions 1 x a where a is the second dimension of v."
  )
  def gradg(p: Tensor) = {
    val tmp0 = ATen.mul_0(p, v.value.value)
    val tmp1 = ATen.sum_1(tmp0, Array(0), false)
    ATen.div_out(tmp1, tmp1, norm)
    tmp0.release
    tmp1
  }

  // https://arxiv.org/pdf/1602.07868.pdf eq3
  // Mind the dot product (.)
  val params = List(
    v.zipBackward { (p, out) =>
      val tmp1 = ATen.div_0(g.value.value, norm)
      val tmp3 = ATen.mul_0(tmp1, p.value)
      val gg = gradg(p.value)
      val tmp2 = ATen.mul_0(g.value.value, gg)
      ATen.div_out(tmp2, tmp2, norm)
      ATen.div_out(tmp2, tmp2, norm)
      val tmp4 = ATen.mul_0(tmp2, v.value.value)
      ATen.add_out(tmp3, tmp3, tmp4, -1d)
      ATen.add_out(out.value, out.value, tmp3, 1d)
      tmp1.release
      tmp2.release
      tmp3.release
      tmp4.release
      gg.release
    },
    g.zipBackward { (p, out) =>
      val tmp2 = gradg(p.value)
      ATen.add_out(out.value, out.value, tmp2, 1d)
      tmp2.release

    }
  )
  // https://arxiv.org/pdf/1602.07868.pdf eq2
  val norm =
    scope(
      ATen.norm_2(
        v.value.value,
        Array(dim),
        false,
        v.options(scope).scalarTypeByte
      )
    )
  val w = ATen.mul_0(v.value.value, g.value.value)
  ATen.div_out(w, w, norm)

  val value = Variable(this, STen.owned(w)(scope))(scope)

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

case class MseLoss(
    scope: Scope,
    input: Variable,
    target: STen,
    reduction: Reduction
) extends Op {

  assert(input.value.numel == target.numel)
  val params = List(
    input.zipBackward { (p, out) =>
      Scope.root { implicit scope =>
        val tmp =
          STen.mse_loss_backward(
            p,
            input.value,
            targetViewed,
            reduction.asLong
          )

        out += tmp
      }

    }
  )
  val targetViewed = target.view(input.shape: _*)(scope)
  val value =
    Variable(
      this,
      STen.mse_loss(input.value, targetViewed, reduction.asLong)(scope)
    )(scope)
}
case class L1Loss(
    scope: Scope,
    input: Variable,
    target: STen,
    reduction: Reduction
) extends Op {
  assert(input.value.numel == target.numel)

  val params = List(
    input.zipBackward { (p, out) =>
      Scope.root { implicit scope =>
        val tmp =
          STen.owned(
            ATen.l1_loss_backward(
              p.value,
              input.value.value,
              targetViewed.value,
              reduction.asLong
            )
          )

        out += tmp

      }
    }
  )
  val targetViewed = target.view(input.shape: _*)(scope)
  val value =
    Variable(
      this,
      STen.owned(
        ATen.l1_loss(input.value.value, targetViewed.value, reduction.asLong)
      )(scope)
    )(scope)
}
case class NllLoss(
    scope: Scope,
    input: Variable,
    target: STen,
    weights: STen,
    // numClasses: Int,
    reduction: Reduction,
    ignore: Long
) extends Op {

  assert(
    input.sizes.size == 2,
    "Nll Loss assumes 2D input (samples x case classes). Higher dimensions not implemented."
  )
  assert(
    target.sizes.size == 1,
    "Target should be a 1D tensor with [0,C-1] integers, C number of case classes."
  )

  val params = List(
    input.zipBackward { (p, out) =>
      Scope.root { implicit scope =>
        val tmp =
          STen.owned(
            ATen.nll_loss_backward(
              p.value,
              input.value.value,
              target.value,
              weights.value,
              reduction.asLong,
              ignore,
              total_weight
            )
          )
        out += tmp

      }
    }
  )
  val (value1, total_weight) =
    ATen.nll_loss_forward(
      input.value.value,
      target.value,
      weights.value,
      reduction.asLong,
      ignore
    )
  scope.register(total_weight)

  val value =
    Variable(
      this,
      STen.owned(value1)(scope)
    )(scope)

}

/** input: (N,T) where T>=1 are multiple independent tasks
  * target: same shape as input, float with in [0,1]
  * posWeight: is (T)
  */
case class BinaryCrossEntropyWithLogitsLoss(
    scope: Scope,
    input: Variable,
    target: STen,
    posWeights: Option[STen],
    reduction: Reduction
) extends Op {

  assert(
    input.sizes == target.sizes,
    s"BinaryCrossEntropyWithLogitsLoss input and target have the same shape. Input ${input.sizes}, target ${target.sizes} ."
  )

  val params = List(
    input.zipBackward { (p, out) =>
      Scope.root { implicit scope =>
        val tmp =
          STen.owned(
            ATen.binary_cross_entropy_with_logits_backward(
              p.value,
              input.value.value,
              target.value,
              None,
              posWeights.map(_.value),
              reduction.asLong
            )
          )
        out += tmp

      }
    }
  )

  val (value1) =
    ATen.binary_cross_entropy_with_logits(
      input.value.value,
      target.value,
      None,
      posWeights.map(_.value),
      reduction.asLong
    )

  val value =
    Variable(
      this,
      STen.owned(value1)(scope)
    )(scope)

}

case class SquaredFrobeniusMatrixNorm(scope: Scope, a: Variable) extends Op {

  val params = List(a.zipBackward { (p, out) =>
    out.addcmulSelf(p, a.value, 2d)
  })
  val value =
    Variable(
      this, {
        val fr = a.value.frobeniusNorm(scope)
        fr.pow_(2d)
        fr
      }
    )(scope)
}

/** 1D convolution
  *
  * @param input batch x in_channels x L
  * @param weight out_channels x in_channels x kernel_size
  * @param bias out_channels
  * @return Variable with Tensor of size batch x out_channels x L' (length depends on stride/padding/dilation)
  */
case class Conv1D(
    scope: Scope,
    input: Variable,
    weight: Variable,
    bias: Variable,
    stride: Long,
    padding: Long,
    dilation: Long,
    groups: Long
) extends Op {

  assert(input.shape.size == 3, "Input dimensions must be 3")
  assert(weight.shape.size == 3, "Weight dimensions must be 3")
  val batchSize = input.shape(0)
  val inputChannels = input.shape(1)
  val imageSize = input.shape(2)
  val kernelSize = weight.shape(2)
  val outChannels = weight.shape(0)
  assert(
    weight.shape(1) == inputChannels,
    "Weight 2nd dimension must have size equal to input channels (2nd dim of input) "
  )
  assert(
    bias.shape(0) == outChannels,
    "Number of biases must be the number of output channels"
  )

  override val params: List[(Variable, (STen, STen) => Unit)] = List(
    input.zipBackward { (p, out) =>
      val pSize = p.sizes
      val zeros = ATen.zeros(Array(inputChannels), p.options(scope).value)
      val outputSizeWithoutExtraPadding =
        (pSize(2) - 1) * stride - 2 * padding + dilation * (kernelSize - 1) + 1
      val extraPadding = out.sizes.apply(2) - outputSizeWithoutExtraPadding
      val tmp = ATen.conv_transpose1d(
        p.value,
        weight.value.value,
        Some(zeros),
        Array(stride),
        Array(padding),
        Array(extraPadding),
        groups,
        Array(dilation)
      )
      ATen.add_out(out.value, out.value, tmp, 1d)
      tmp.release
      zeros.release()
    },
    weight.zipBackward { (p, out) =>
      val p_repeated =
        ATen.repeat_interleave_2(p.value, inputChannels / groups, 1)
      val p_repeated_size = p_repeated.sizes
      val p_repeated_viewed =
        ATen._unsafe_view(
          p_repeated,
          Array(p_repeated_size(0) * p_repeated_size(1), 1, p_repeated_size(2))
        )
      val input_viewed = ATen._unsafe_view(
        input.value.value,
        Array(1, batchSize * inputChannels, imageSize)
      )
      val zero = ATen
        .zeros(Array(p_repeated_viewed.sizes.apply(0)), p.options(scope).value)
      val conv_0 = ATen.conv1d(
        input_viewed,
        p_repeated_viewed,
        Some(zero),
        Array(dilation),
        Array(padding),
        Array(stride),
        inputChannels * batchSize
      )
      val conv_0_sizes = conv_0.sizes
      val conv_1 = ATen._unsafe_view(
        conv_0,
        Array(
          batchSize,
          conv_0_sizes.apply(1) / batchSize,
          conv_0_sizes.apply(2)
        )
      )

      val conv_1_sum = ATen.sum_1(conv_1, Array(0L), false)
      val conv_1_sum_viewed =
        ATen._unsafe_view(
          conv_1_sum,
          Array(inputChannels / groups, outChannels, conv_1.sizes.apply(2))
        )
      val conv_1_sum_viewed_transposed = ATen.transpose(conv_1_sum_viewed, 0, 1)

      val conv_1_sum_viewed_transposed_narrowed =
        ATen.narrow_0(conv_1_sum_viewed_transposed, 2, 0, kernelSize)
      ATen.add_out(
        out.value,
        out.value,
        conv_1_sum_viewed_transposed_narrowed,
        1d
      )

      conv_1_sum_viewed_transposed_narrowed.release()
      conv_1_sum_viewed_transposed.release
      conv_1_sum_viewed.release
      conv_1_sum.release
      conv_1.release
      conv_0.release
      input_viewed.release()
      p_repeated_viewed.release
      p_repeated.release

    },
    bias.zipBackward { (p, out) =>
      val p2 = ub(p.value, List(out.sizes.toList.head, 1)).getOrElse(p.value)
      val p3 = ATen._unsafe_view(p2, out.sizes.toArray)
      ATen.add_out(out.value, out.value, p3, 1d)
      if (p2 != p.value) {
        p2.release()
      }
      p3.release()
    }
  )

  val value =
    Variable(
      this, {
        STen.owned(
          ATen.conv1d(
            input.value.value,
            weight.value.value,
            Some(bias.value.value),
            Array(stride),
            Array(padding),
            Array(dilation),
            groups
          )
        )(scope)
      }
    )(scope)
}

/** 2D convolution
  *
  * @param input batch x in_channels x height x width
  * @param weight out_channels x in_channels x kernel_size x kernel_size
  * @param bias out_channels
  * @return Variable with Tensor of size batch x out_channels x L' (length depends on stride/padding/dilation)
  */
case class Conv2D(
    scope: Scope,
    input: Variable,
    weight: Variable,
    bias: Variable,
    stride: Long,
    padding: Long,
    dilation: Long,
    groups: Long
) extends Op {

  assert(input.shape.size == 4, "Input dimensions must be 4")
  assert(weight.shape.size == 4, "Weight dimensions must be 4")
  val batchSize = input.shape(0)
  val inputChannels = input.shape(1)
  val imageHeight = input.shape(2)
  val imageWidth = input.shape(3)
  val kernelSize = weight.shape(2)
  val outChannels = weight.shape(0)

  override val params: List[(Variable, (STen, STen) => Unit)] = List(
    input.zipBackward { (p, out) =>
      if (p.options(scope).value.isCuda()) {
        val tmp = ATen.cudnn_convolution_backward_input(
          input.shape.toArray,
          p.value,
          weight.value.value,
          Array(padding, padding),
          Array(stride, stride),
          Array(dilation, dilation),
          groups,
          false,
          false,
          true
        )
        ATen.add_out(out.value, out.value, tmp, 1d)
        tmp.release
      } else {
        val pSize = p.sizes
        val zeros = ATen.zeros(Array(inputChannels), p.options(scope).value)
        val outputSizeWithoutExtraPadding =
          (pSize(
            2
          ) - 1) * stride - 2 * padding + dilation * (kernelSize - 1) + 1
        val extraPaddingH = out.sizes.apply(2) - outputSizeWithoutExtraPadding
        val extraPaddingW = out.sizes.apply(3) - outputSizeWithoutExtraPadding
        val tmp = ATen.conv_transpose2d(
          p.value,
          weight.value.value,
          Some(zeros),
          Array(stride),
          Array(padding),
          Array(extraPaddingH, extraPaddingW),
          groups,
          Array(dilation)
        )
        ATen.add_out(out.value, out.value, tmp, 1d)
        tmp.release
        zeros.release()
      }
    },
    weight.zipBackward { (p, out) =>
      if (p.options(scope).value.isCuda()) {
        val tmp = ATen.cudnn_convolution_backward_weight(
          weight.shape.toArray,
          p.value,
          input.value.value,
          Array(padding, padding),
          Array(stride, stride),
          Array(dilation, dilation),
          groups,
          false,
          false,
          true
        )
        ATen.add_out(out.value, out.value, tmp, 1d)
        tmp.release
      } else {
        if (groups == 1) {

          val (a, b, c) = ATen.slow_conv_dilated2d_backward(
            p.value,
            input.value.value,
            weight.value.value,
            Array(kernelSize, kernelSize),
            Array(stride, stride),
            Array(padding, padding),
            Array(dilation, dilation),
            Array(false, true, false)
          )

          a.release
          c.release
          ATen.add_out(out.value, out.value, b, 1d)
          b.release

        } else {
          throw new NotImplementedError(
            "Conv2d backward with groups>1 on cpu is not implemented"
          )
          val p_repeated =
            ATen.repeat_interleave_2(p.value, inputChannels / groups, 1)
          val p_repeated_size = p_repeated.sizes
          val p_repeated_viewed =
            ATen._unsafe_view(
              p_repeated,
              Array(
                p_repeated_size(0) * p_repeated_size(1),
                1,
                p_repeated_size(2),
                p_repeated_size(3)
              )
            )
          val input_viewed = ATen._unsafe_view(
            input.value.value,
            Array(1, batchSize * inputChannels, imageHeight, imageWidth)
          )
          val zero =
            ATen.zeros(
              Array(p_repeated_viewed.sizes.apply(0)),
              p.options(scope).value
            )
          val conv_0 = ATen.conv2d(
            input_viewed,
            p_repeated_viewed,
            Some(zero),
            Array(dilation),
            Array(padding),
            Array(stride),
            inputChannels * batchSize
          )
          val conv_0_sizes = conv_0.sizes
          val conv_1 = ATen._unsafe_view(
            conv_0,
            Array(
              batchSize,
              conv_0_sizes.apply(1) / batchSize,
              conv_0_sizes.apply(2),
              conv_0_sizes.apply(3)
            )
          )
          val conv_1_sum = ATen.sum_1(conv_1, Array(0L), false)
          val conv_1_sum_viewed =
            ATen._unsafe_view(
              conv_1_sum,
              Array(
                inputChannels / groups,
                outChannels,
                conv_1.sizes.apply(2),
                conv_1.sizes.apply(3)
              )
            )
          val conv_1_sum_viewed_transposed =
            ATen.transpose(conv_1_sum_viewed, 0, 1)

          val conv_1_sum_viewed_transposed_narrowed1 =
            ATen.narrow_0(conv_1_sum_viewed_transposed, 2, 0, kernelSize)
          val conv_1_sum_viewed_transposed_narrowed2 =
            ATen
              .narrow_0(
                conv_1_sum_viewed_transposed_narrowed1,
                3,
                0,
                kernelSize
              )

          ATen.add_out(
            out.value,
            out.value,
            conv_1_sum_viewed_transposed_narrowed2,
            1d
          )

          conv_1_sum_viewed_transposed_narrowed1.release()
          conv_1_sum_viewed_transposed_narrowed2.release()
          conv_1_sum_viewed_transposed.release
          conv_1_sum_viewed.release
          conv_1_sum.release
          conv_1.release
          conv_0.release
          input_viewed.release()
          p_repeated_viewed.release
          p_repeated.release
          zero.release
        }
      }
    },
    bias.zipBackward { (p, out) =>
      val p2 = ub(p.value, List(out.sizes.toList.head, 1, 1)).getOrElse(p.value)
      val p3 = ATen._unsafe_view(p2, out.sizes.toArray)

      ATen.add_out(out.value, out.value, p3, 1d)
      if (p2 != p.value) {
        p2.release()
      }
      p3.release()
    }
  )

  val value =
    Variable(
      this, {
        STen.owned(
          ATen.conv2d(
            input.value.value,
            weight.value.value,
            Some(bias.value.value),
            Array(stride),
            Array(padding),
            Array(dilation),
            groups
          )
        )(scope)
      }
    )(scope)
}

case class Conv2DTransposed(
    scope: Scope,
    input: Variable,
    weight: Variable,
    bias: Variable,
    stride: Long,
    padding: Long,
    dilation: Long
    // groups: Long
) extends Op {

  assert(
    input.shape.size == 4,
    s"Input dimensions must be 4, got ${input.shape}"
  )
  assert(
    weight.shape.size == 4,
    s"Weight dimensions must be 4, got ${weight.shape}"
  )
  val batchSize = input.shape(0)
  val inputChannels = input.shape(1)
  val imageHeight = input.shape(2)
  val imageWidth = input.shape(3)
  val kernelSize = weight.shape(2)
  val outChannels = weight.shape(1)

  override val params: List[(Variable, (STen, STen) => Unit)] = List(
    input.zipBackward { (p, out) =>
      if (p.scalarTypeByte == 6 && p.options(scope).value.isCuda()) {
        val tmp = ATen.cudnn_convolution_transpose_backward_input(
          p.value,
          weight.value.value,
          Array(padding, padding),
          Array(stride, stride),
          Array(dilation, dilation),
          1,
          false,
          false,
          true
        )
        ATen.add_out(out.value, out.value, tmp, 1d)
        tmp.release
      } else {

        val columns = STen.zerosLike(p)(scope)
        val ones = STen.onesLike(p)(scope)
        assert(input.shape(1) == weight.shape(0))
        assert(p.shape(1) == weight.shape(1))
        assert(p.shape(1) == bias.shape(0))

        assert(
          p.shape(2) == ((imageWidth - 1) * stride - 2 * padding +
            (dilation * (kernelSize - 1) + 1))
        )
        assert(
          p.shape(3) == ((imageHeight - 1) * stride - 2 * padding +
            (dilation * (kernelSize - 1) + 1))
        )
        val (a, b, c) = ATen.slow_conv_transpose2d_backward(
          p.value,
          input.value.value,
          weight.value.value,
          Array(kernelSize, kernelSize),
          Array(stride, stride),
          Array(padding, padding),
          Array(0, 0),
          Array(dilation, dilation),
          columns.value,
          ones.value,
          Array(true, false, false)
        )

        b.release
        c.release
        ATen.add_out(out.value, out.value, a, 1d)
        a.release

      }
    },
    weight.zipBackward { (p, out) =>
      if (p.scalarTypeByte == 6 && p.options(scope).value.isCuda()) {
        val tmp = ATen.cudnn_convolution_transpose_backward_weight(
          weight.shape.toArray,
          p.value,
          input.value.value,
          Array(padding, padding),
          Array(stride, stride),
          Array(dilation, dilation),
          1,
          false,
          false,
          true
        )
        ATen.add_out(out.value, out.value, tmp, 1d)
        tmp.release
      } else {
        val columns = STen.zerosLike(p)(scope)
        val ones = STen.onesLike(p)(scope)
        val (a, b, c) = ATen.slow_conv_transpose2d_backward(
          p.value,
          input.value.value,
          weight.value.value,
          Array(kernelSize, kernelSize),
          Array(stride, stride),
          Array(padding, padding),
          Array(0, 0),
          Array(dilation, dilation),
          columns.value,
          ones.value,
          Array(false, true, false)
        )

        a.release
        c.release
        ATen.add_out(out.value, out.value, b, 1d)
        b.release
      }
    },
    bias.zipBackward { (p, out) =>
      val columns = STen.zerosLike(p)(scope)
      val ones = STen.onesLike(p)(scope)
      val (a, b, c) = ATen.slow_conv_transpose2d_backward(
        p.value,
        input.value.value,
        weight.value.value,
        Array(kernelSize, kernelSize),
        Array(stride, stride),
        Array(padding, padding),
        Array(0, 0),
        Array(dilation, dilation),
        columns.value,
        ones.value,
        Array(false, false, true)
      )

      b.release
      a.release
      ATen.add_out(out.value, out.value, c, 1d)
      c.release
    }
  )

  val value =
    Variable(
      this, {
        STen.owned(
          ATen.conv_transpose2d(
            input.value.value,
            weight.value.value,
            Some(bias.value.value),
            Array(stride, stride),
            Array(padding, padding),
            Array(0, 0),
            1,
            Array(dilation)
          )
        )(scope)
      }
    )(scope)
}

/** 1D max pooling
  *
  * @param input batch x in_channels x L
  */
case class MaxPool1D(
    scope: Scope,
    input: Variable,
    kernelSize: Long,
    stride: Long = 1,
    padding: Long = 0,
    dilation: Long = 1
) extends Op {

  assert(input.shape.size == 3, "Input dimensions must be 3")
  val batchSize = input.shape(0)
  val inputChannels = input.shape(1)
  val imageSize = input.shape(2)

  override val params: List[(Variable, (STen, STen) => Unit)] = List(
    input.zipBackward { (p, out) =>
      val zeros = ATen.zeros_like(out.value, input.options(scope).value)
      val p_flatten = ATen.flatten(p.value, 0, 1)
      val mask_flatten = ATen.flatten(mask, 0, 1)
      val zeros_flatten = ATen.flatten(zeros, 0, 1)
      val addeds = 0L until p_flatten.shape(0) map { i =>
        val p_select = ATen.select(p_flatten, 0, i)
        val mask_select = ATen.select(mask_flatten, 0, i)
        val zeros_select = ATen.select(zeros_flatten, 0, i)
        val added = ATen.index_add(zeros_select, 0, mask_select, p_select)
        p_select.release
        mask_select.release
        zeros_select.release
        added
      }

      val catted = ATen.cat(addeds.toArray, 0)
      addeds.foreach(_.release)
      val catted_viewed = ATen._unsafe_view(catted, out.sizes.toArray)
      ATen.add_out(out.value, out.value, catted_viewed, 1d)

      catted_viewed.release
      catted.release
      zeros_flatten.release
      mask_flatten.release
      p_flatten.release
      zeros.release
    }
  )

  val (output, mask) = ATen.max_pool1d_with_indices(
    input.value.value,
    Array(kernelSize),
    Array(stride),
    Array(padding),
    Array(dilation),
    false
  )
  scope.register(mask)
  val value =
    Variable(this, STen.owned(output)(scope))(scope)
}

/** 2D max pooling
  *
  * @param input batch x in_channels x h x w
  */
case class MaxPool2D(
    scope: Scope,
    input: Variable,
    kernelSize: Long,
    stride: Long,
    padding: Long,
    dilation: Long
) extends Op {

  assert(input.shape.size == 4, "Input dimensions must be 4")
  val batchSize = input.shape(0)
  val inputChannels = input.shape(1)
  val imageSize = input.shape(2)

  override val params: List[(Variable, (STen, STen) => Unit)] = List(
    input.zipBackward { (p, out) =>
      Scope.root { implicit scope =>
        val tmp = STen.owned(
          ATen.max_pool2d_with_indices_backward(
            p.value,
            input.value.value,
            Array(kernelSize),
            Array(stride),
            Array(padding),
            Array(dilation),
            false,
            mask
          )
        )
        out += tmp
      }

    }
  )

  val (output, mask) = ATen.max_pool2d_with_indices(
    input.value.value,
    Array(kernelSize),
    Array(stride),
    Array(padding),
    Array(dilation),
    false
  )
  scope.register(mask)

  val value =
    Variable(this, STen.owned(output)(scope))(scope)
}

/** 2D avg pooling
  *
  * @param input batch x in_channels x h x w
  */
case class AvgPool2D(
    scope: Scope,
    input: Variable,
    kernelSize: Long,
    stride: Long,
    padding: Long
) extends Op {

  assert(input.shape.size == 4, "Input dimensions must be 4")
  val batchSize = input.shape(0)
  val inputChannels = input.shape(1)
  val imageSize = input.shape(2)

  override val params: List[(Variable, (STen, STen) => Unit)] = List(
    input.zipBackward { (p, out) =>
      Scope.root { implicit scope =>
        val tmp = STen.owned(
          ATen.avg_pool2d_backward(
            p.value,
            input.value.value,
            Array(kernelSize),
            Array(stride),
            Array(padding),
            false,
            true,
            Long.MinValue
          )
        )

        out += tmp
      }

    }
  )

  val value =
    Variable(
      this,
      STen.owned(
        ATen.avg_pool2d(
          input.value.value,
          Array(kernelSize),
          Array(stride),
          Array(padding),
          false,
          true,
          Long.MinValue
        )
      )(scope)
    )(scope)
}

case class Flatten(scope: Scope, input: Variable, startDim: Int, endDim: Int)
    extends Op {

  assert(input.shape.size >= 2, "Input dimensions must be 4")

  override val params: List[(Variable, (STen, STen) => Unit)] = List(
    input.zipBackward { (p, out) =>
      Scope.root { implicit scope => out += p.view(out.shape: _*) }
    }
  )

  val value =
    Variable(
      this,
      input.value.flatten(startDim, endDim)(scope)
    )(scope)
}

/* 0-th dimension has samples. Everything else is flattened out into features. */
case class BatchNorm(
    scope: Scope,
    input: Variable,
    weight: Variable,
    bias: Variable,
    runningMean: STen,
    runningVar: STen,
    training: Boolean,
    momentum: Double,
    eps: Double
) extends Op {

  val input_flattened = ATen.flatten(input.value.value, 1, input.shape.size - 1)
  val expectedShape = List(input_flattened.shape.last)
  assert(
    expectedShape == weight.shape,
    s"Expected $expectedShape got weight shape ${weight.shape}"
  )
  assert(
    expectedShape == bias.shape,
    s"Expected $expectedShape got bias shape ${bias.shape}"
  )
  assert(
    expectedShape == runningMean.shape,
    s"Expected $expectedShape got runningMean shape ${runningMean.shape}"
  )
  assert(
    expectedShape == runningVar.shape,
    s"Expected $expectedShape got runningVar shape ${runningVar.shape}"
  )

  val (output, saveMean, saveInvstd) = ATen.native_batch_norm(
    input_flattened,
    weight.value.value,
    bias.value.value,
    runningMean.value,
    runningVar.value,
    training,
    momentum,
    eps
  )
  val output_reshaped = ATen._unsafe_view(output, input.shape.toArray)

  scope.register(saveMean)
  scope.register(saveInvstd)
  scope.register(input_flattened)
  scope.register(output)

  override val value: Variable =
    Variable(this, STen.owned(output_reshaped)(scope))(scope)

  override val params: List[(Variable, (STen, STen) => Unit)] = List(
    input.zipBackward { (p, out) =>
      val flattened_p =
        ATen.flatten(p.value, 1, p.shape.size - 1)
      val (gradInput, a, b) = ATen.native_batch_norm_backward(
        flattened_p,
        input_flattened,
        weight.value.value,
        runningMean.value,
        runningVar.value,
        saveMean,
        saveInvstd,
        training,
        eps,
        Array(true, false, false)
      )
      val gradInput_reshaped = ATen._unsafe_view(gradInput, out.shape.toArray)
      ATen.add_out(out.value, out.value, gradInput_reshaped, 1d)
      gradInput.release
      a.release
      b.release
      flattened_p.release()
      gradInput_reshaped.release
    },
    weight.zipBackward { (p, out) =>
      val flattened_p =
        ATen.flatten(p.value, 1, p.shape.size - 1)
      val (a, gradWeight, b) = ATen.native_batch_norm_backward(
        flattened_p,
        input_flattened,
        weight.value.value,
        runningMean.value,
        runningVar.value,
        saveMean,
        saveInvstd,
        training,
        eps,
        Array(false, true, false)
      )
      val grad_reshaped = ATen._unsafe_view(gradWeight, out.shape.toArray)
      ATen.add_out(out.value, out.value, grad_reshaped, 1d)
      gradWeight.release
      grad_reshaped.release
      a.release
      b.release
      flattened_p.release()
    },
    bias.zipBackward { (p, out) =>
      val flattened_p =
        ATen.flatten(p.value, 1, p.shape.size - 1)
      val tmp = ub(flattened_p, out.shape).getOrElse(flattened_p)
      ATen.add_out(out.value, out.value, tmp, 1d)
      if (tmp != flattened_p) {
        tmp.release
      }
      flattened_p.release
    }
  )
}

/** Batch Norm 2D
  * 0-th dimension are samples. 1-th are features, everything else is averaged out.
  */
case class BatchNorm2D(
    scope: Scope,
    input: Variable,
    weight: Variable,
    bias: Variable,
    runningMean: STen,
    runningVar: STen,
    training: Boolean,
    momentum: Double,
    eps: Double
) extends Op {

  val inputShape = input.shape
  assert(inputShape.size >= 3, "Expected 3D or 4D tensor")
  val expectedShape = List(inputShape(1))
  assert(
    expectedShape == weight.shape,
    s"Expected $expectedShape got weight shape ${weight.shape}"
  )
  assert(
    expectedShape == bias.shape,
    s"Expected $expectedShape got bias shape ${bias.shape}"
  )
  assert(
    expectedShape == runningMean.shape,
    s"Expected $expectedShape got runningMean shape ${runningMean.shape}"
  )
  assert(
    expectedShape == runningVar.shape,
    s"Expected $expectedShape got runningVar shape ${runningVar.shape}"
  )

  val (output, saveMean, saveInvstd) = ATen.native_batch_norm(
    input.value.value,
    weight.value.value,
    bias.value.value,
    runningMean.value,
    runningVar.value,
    training,
    momentum,
    eps
  )
  scope.register(saveMean)
  scope.register(saveInvstd)
  override val value: Variable =
    Variable(this, STen.owned(output)(scope))(scope)

  override val params: List[(Variable, (STen, STen) => Unit)] = List(
    input.zipBackward { (p, out) =>
      val (gradInput, a, b) = ATen.native_batch_norm_backward(
        p.value,
        input.value.value,
        weight.value.value,
        runningMean.value,
        runningVar.value,
        saveMean,
        saveInvstd,
        training,
        eps,
        Array(true, false, false)
      )
      val gradInput_reshaped =
        ATen._unsafe_view(gradInput, out.shape.toArray)
      ATen.add_out(out.value, out.value, gradInput_reshaped, 1d)
      gradInput.release
      a.release
      b.release
      gradInput_reshaped.release
    },
    weight.zipBackward { (p, out) =>
      val (a, gradWeight, b) = ATen.native_batch_norm_backward(
        p.value,
        input.value.value,
        weight.value.value,
        runningMean.value,
        runningVar.value,
        saveMean,
        saveInvstd,
        training,
        eps,
        Array(false, true, false)
      )
      val grad_reshaped = ATen._unsafe_view(gradWeight, out.shape.toArray)
      ATen.add_out(out.value, out.value, grad_reshaped, 1d)
      gradWeight.release
      grad_reshaped.release
      a.release
      b.release
    },
    bias.zipBackward { (p, out) =>
      val tmp = ub(p.value, (out.shape ++ (1 to p.shape.size - 2).map(_ => 1L)))
        .getOrElse(p.value)
      val tmp_viewed = ATen._unsafe_view(
        tmp,
        out.shape.toArray
      )
      ATen.add_out(out.value, out.value, tmp_viewed, 1d)
      if (p.value != tmp) {
        tmp.release
      }
      tmp_viewed.release
    }
  )
}
case class Embedding(scope: Scope, input: Variable, weight: Variable)
    extends Op {

  assert(input.shape.size >= 1, "Input dimensions must be at least 1")
  assert(weight.shape.size == 2, "Weight must have 2 dimensions")
  override val params: List[(Variable, (STen, STen) => Unit)] = List(
    weight.zipBackward { (p, out) =>
      Scope.root { implicit scope =>
        val tmp = STen.owned(
          ATen
            .embedding_backward(
              p.value,
              input.value.value,
              weight.shape(0),
              0L,
              false,
              false
            )
        )
        out += tmp

      }

    },
    input.zipBackward { (_, _) => () }
  )

  val value =
    try {
      Variable(
        this,
        STen.owned(
          ATen
            .embedding(weight.value.value, input.value.value, 0L, false, false)
        )(scope)
      )(scope)
    } catch {
      case e: Throwable =>
        println(weight.shape)
        println(input.value.toLongMat)
        throw e
    }
}

case class Cholesky(
    scope: Scope,
    input: Variable,
    upper: Boolean
) extends Op {

  val params = List(
    input.zipBackward { case (p, out) =>
      Scope.root { implicit scope =>
        // Ref: Pytorch FunctionsManual.cpp
        // https://arxiv.org/pdf/1602.07527.pdf
        val batch = input.shape.size == 3
        val size = input.shape(input.shape.size - 2)
        val l =
          if (!upper) value.value
          else value.value.transpose(-1, -2)

        val g =
          if (!upper) p
          else p.transpose(-1, -2)

        val lInv =
          STen
            .triangularSolve(
              STen.eye(size.toInt, l.options),
              l,
              false,
              false,
              false
            )
        val phi =
          if (batch) l.transpose(-1, -2).bmm(g)
          else l.transpose(-1, -2).mm(g)
        phi.tril_()
        phi.diagonalView(0, -2, -1) *= 0.5

        val tmp =
          if (batch)
            lInv.transpose(-1, -2).bmm(phi).bmm(lInv)
          else lInv.transpose(-1, -2).mm(phi).mm(lInv)

        val tmp2 = tmp + tmp.transpose(-1, -2)
        tmp2 *= 0.5
        out += tmp2

      }
    }
  )
  val value = Variable(
    this,
    input.value.cholesky(upper)(scope)
  )(scope)
}
case class CholeskySolve(
    scope: Scope,
    b: Variable,
    factor: Variable,
    upper: Boolean
) extends Op {

  val params = List(
    b.zipBackward { case (p, out) =>
      Scope.root { implicit scope =>
        val gB = p.choleskySolve(factor.value, upper)

        out += gB

      }
    },
    factor.zipBackward { case (p, out) =>
      Scope.root { implicit scope =>
        val gB = p.choleskySolve(factor.value, upper)
        val batched = b.shape.size == 3
        val tmp =
          if (batched) gB.bmm(value.value.transpose(-2, -1))
          else gB.mm(value.value.transpose(-2, -1))

        val tmp2 = tmp + tmp.transpose(-2, -1)

        val tmp4 = if (upper) {
          if (batched) factor.value.bmm(tmp2)
          else factor.value.mm(tmp2)
        } else {
          if (batched) tmp2.bmm(factor.value)
          else tmp2.mm(factor.value)
        }

        // symmetrize it
        tmp4.tril_()
        tmp4.diagonalView(0, -2, -1) *= 0.5

        val symmetrized = tmp4 + tmp4.transpose(-2, -1)

        out -= symmetrized

      }
    }
  )
  val value = Variable(
    this,
    b.value.choleskySolve(factor.value, upper)(
      scope
    )
  )(scope)
}
