package lamp.autograd
// import aten.Tensor
// import aten.ATen
import lamp.FloatingPointPrecision
import lamp.DoublePrecision
import lamp.SinglePrecision

import lamp.Scope
import lamp.STen
import lamp.Bf16Precision
import lamp.HalfPrecision
// import lamp.STenOptions


case class Persist(unpersist: Variable) extends Op {

  val params = List(
    unpersist.zipBackward { case Grad(p, out) =>
      Scope.root {  _ => out += p }

    }
  )
  val value = Variable.persist(
    this,
    _ => implicit forward => unpersist.forward
  )
}
case class Transpose(a: Variable, dim1: Int = 0, dim2: Int = 1) extends Op {

  val params = List(
    a.zipBackward { case Grad(p, out) =>
      Scope.root { implicit scope => out += p.transpose(dim1, dim2) }

    }
  )
  val value = Variable(
    this,
    implicit scope => implicit forward => a.forward.transpose(dim1, dim2)
  )
}

case class View(a: Variable, shape: List[Long]) extends Op {

  val params = List(
    a.zipBackward { case Grad(p, out) =>
      Scope.root { implicit scope => out += p.view(out.shape: _*) }

    }
  )
  val value =
    Variable(
      this,
      { implicit scope => implicit forward =>
        a.forward.view(shape: _*)
      }
    )
}
case class Reshape(a: Variable, shape: List[Long]) extends Op {

  val params = List(
    a.zipBackward { case Grad(p, out) =>
      Scope.root { implicit scope => out += p.reshape(out.shape: _*) }

    }
  )
  val value = Variable(
    this,
    implicit scope =>
      implicit forward =>
        a.forward.reshape(shape: _*)
  )
}

case class Concatenate(a: Seq[Variable], dim: Long) extends Op {
  val params = a.toList.zipWithIndex.map { case (input, idx) =>
    input.zipBackward { case grad @ Grad(p, out) =>
      implicit val fw = grad.fw
      Scope.root { implicit scope =>
      val ashapes = a.map(_.shape(fw)(dim.toInt))
      val boundaries =
        ashapes.scanLeft(0L)(_ + _).sliding(2).toList.map(g => g(0) -> g(1))

      val (from, to) = boundaries(idx)
       out += p.slice(dim, from, to, 1L) }
    }
  }
  val value =
    Variable(
      this,
      implicit scope => implicit forward => STen.cat(a.map(_.forward), dim)
    )
}

case class Stack(a: Seq[Variable], dim: Long) extends Op {
  val params = a.zipWithIndex.toList.map { case (a, idx) =>
    a.zipBackward { case Grad(p, out) =>
      Scope.root { implicit scope => out += p.select(dim, idx) }
    }
  }
  val value =
    Variable(
      this,
      implicit scope => implicit forward => STen.stack(a.map(_.forward), dim)
    )
}

case class Select(a: Variable, dim: Long, index: Long) extends Op {

  val params = List(
    a.zipBackward { case grad @ Grad(p, out) =>
      implicit val fw = grad.fw
      Scope.root { implicit scope =>
        val av = value.forward
        val tmp = STen.zeros(out.sizes, av.options)
        val scalar = STen.scalarLong(index, av.options)
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
    Variable(
      this,
      implicit scope => implicit forward => a.forward.select(dim, index)
    )
}
case class Slice(
    a: Variable,
    dim: Long,
    start: Long,
    end: Long,
    step: Long
) extends Op {

  val params = List(
    a.zipBackward { case Grad(p, out) =>
      Scope.root { implicit scope =>
        val tmp = STen.zeros(out.sizes, out.options)
        val index = STen.arange_l(start, end, step, out.options.toLong)
        val tmp2 = tmp.indexAdd(dim, index, p)
        out += tmp2

      }
    }
  )
  val value =
    Variable(
      this,
      implicit scope =>
        implicit forward => a.forward.slice(dim, start, end, step)
    )
}
case class EqWhere(a: Variable, b: Long) extends Op {

  val params = List()
  val value = Variable(
    this,
    implicit scope => implicit forward => a.forward.equ(b)
  )
}
case class MaskSelect(input: Variable, mask: Variable) extends Op {

  val params = List(
    input.zipBackward { case grad =>
      implicit val fw = grad.fw
      Scope.root { implicit scope =>
        val tmp = STen.zerosLike(grad.out)
        grad.out += tmp.maskedScatter(mask.forward, grad.p)
      }

    }
  )
  val value =
    Variable(
      this,
      implicit scope =>
        implicit forward => input.forward.maskedSelect(mask.forward)
    )
}
case class MaskFill(input: Variable, mask: STen, fill: Double) extends Op {

  val params = List(
    input.zipBackward { case grad =>
      Scope.root { implicit scope =>
        grad.out += grad.p.maskedFill(mask, 0d)
      }

    }
  )
  val value =
    Variable(
      this,
      implicit scope =>
        implicit forward => input.forward.maskFill(mask, fill)
    )
}
case class IndexFill(
    input: Variable,
    dim: Long,
    index: STen,
    fill: Double
) extends Op {

  val params = List(
    input.zipBackward { case grad =>
      Scope.root { implicit scope =>
        grad.out += grad.p.indexFill(dim, index, 0d)
      }

    }
  )
  val value =
    Variable(
      this,
      implicit scope =>
        implicit forward => input.forward.indexFill(dim, index, fill)
    )
}
case class IndexSelect(
    input: Variable,
    dim: Long,
    index: STen
) extends Op {

  val params = List(
    input.zipBackward { case grad =>
      Scope.root { implicit scope =>
        val tmp = grad.out.indexAdd(dim, index, grad.p)
        grad.out += tmp
      }

    }
  )
  val value =
    Variable(
      this,
      implicit scope =>
        implicit forward => input.forward.indexSelect(dim, index)
    )
}
case class Where(
    condition: STen,
    trueBranch: Variable,
    falseBranch: Variable
) extends Op {

  val params = List(
    trueBranch.zipBackward { case grad @ Grad(p, out) =>
      implicit val fw = grad.fw
      Scope.root { implicit scope =>
        val ones = STen.onesLike(trueBranch.forward)
        val zeros = STen.zerosLike(falseBranch.forward)
        val tmp = STen.where(condition, ones, zeros)
        out.addcmulSelf(p, tmp, 1.0)
      }
    },
    falseBranch.zipBackward { case grad @ Grad(p, out) =>
      implicit val fw = grad.fw
      Scope.root { implicit scope =>
        val zeros = STen.zerosLike(trueBranch.forward)
        val ones = STen.onesLike(falseBranch.forward)
        val tmp = STen.where(condition, zeros, ones)
        out.addcmulSelf(p, tmp, 1.0)
      }

    }
  )
  val value =
    Variable(
      this,
      implicit scope =>
        implicit forward =>
          STen.where(
            condition,
            trueBranch.forward,
            falseBranch.forward
          )
    )
}
case class ArgMax(a: Variable, dim: Long, keepDim: Boolean) extends Op {

  val params = List(
    a.missingBackward
  )
  val value =
    Variable(
      this,
      implicit scope => implicit forward => a.forward.argmax(dim, keepDim)
    )
}

/** x = a.assign(b) x's value is b, x's grad is 1 wrt b and 0 wrt to a
  */
case class Assign(abandon: Variable, keep: Variable) extends Op {

  val params = List(
    abandon.zipBackward { case _ => () },
    keep.zipBackward { case Grad(p, out) => out += p }
  )
  val value = Variable(this, _ => implicit forward => keep.forward)
}
case class OneHot(a: Variable, numClasses: Int) extends Op {

  val params = List(
    a.missingBackward
  )
  val value =
    Variable(
      this,
      implicit scope => implicit forward => a.forward.oneHot(numClasses)
    )
}
case class CastToPrecision(
    a: Variable,
    precision: FloatingPointPrecision
) extends Op {

  val params = List(
    a.zipBackward { case grad =>
      implicit val fw = grad.fw
      Scope.root { implicit scope =>
        val st = value.forward.scalarTypeByte
        grad.out += (grad.p.castToType(st))
      }
    }
  )

  val value =
    Variable(
      this,
      { implicit scope => implicit forward =>
        
        val v = a.forward
        val vSt = v.scalarTypeByte
        val vv = precision match {
          case x if x.scalarTypeByte == vSt => v.cloneTensor
          case DoublePrecision              => v.castToDouble
          case SinglePrecision              => v.castToFloat
          case HalfPrecision                => v.castToHalf
          case Bf16Precision                => v.copyTo(v.options.toBF16)
        }
        vv  
      }
    )
}
// case class SparseFromValueAndIndex(
//     values: Variable,
//     indices: STen,
//     dim: Seq[Long]
// ) extends Op {

//   val params = List(
//     values.zipBackward { case Grad(p, out) =>
//       Scope.root { scope =>
//         out += p.values(scope)
//       }
//     }
//   )
//   val value = Variable(
//     this,
//     { implicit scope => implicit forward =>
//       val v = values.forward
//       STen.sparse_coo(
//         indices = indices,
//         values = v,
//         dim = dim,
//         v.device.to(
//           STenOptions
//             .fromScalarType(v.scalarTypeByte)
//         )
//       )
//     }
//   )

// }
case class ToDense(
    sparse: Variable
) extends Op {

  val params = List(
    sparse.zipBackward { case grad @ Grad(p, out) =>
      implicit val fw = grad.fw
      Scope.root { implicit scope =>
        out += STen.to_dense_backward(p, sparse.forward)
      }
    }
  )
  val value = Variable(
    this,
    implicit scope => implicit forward => sparse.forward.toDense
  )
}
case class Diag(
    a: Variable,
    diagonal: Long
) extends Op {

  val params = List(
    a.zipBackward { case Grad(p, out) =>
      Scope.root { implicit scope =>
        out += p.diag(diagonal)
      }
    }
  )
  val value = Variable(
    this,
    implicit scope => implicit forward => a.forward.diag(diagonal)
  )
}
case class Inv(
    a: Variable
) extends Op {
  val params = List(
    a.zipBackward { case grad =>
      implicit val fw = grad.fw
      Scope.root { implicit scope =>
        val v = value.forward
        if (v.shape.size == 3) {
          grad.out -= v.bmm(grad.p).bmm(v).transpose(1, 2)
        } else {
          grad.out -= v.mm(grad.p).mm(v).t
        }
      }
    }
  )
  val value = Variable(
    this,
    implicit scope => implicit forward => a.forward.inv
  )
}
case class PInv(
    a: Variable,
    rcond: Double
) extends Op {
  val params = List(
    a.zipBackward { case grad =>
      implicit val fw = grad.fw
      Scope.root { implicit scope =>
        val v = value.forward
        val p = grad.p
        val out = grad.out
        val avalue = a.forward
        // https://math.stackexchange.com/questions/2179160/derivative-of-pseudoinverse-with-respect-to-original-matrix
        if (v.shape.size == 3) {
          val vt = v.transpose(1, 2)
          val pt = p.transpose(1, 2)
          val i = STen.eye(avalue.shape(1).toInt, v.options).unsqueeze(0)

          out += v.bmm(vt).bmm(pt).bmm(i - avalue.bmm(v)) + (i - v
            .bmm(avalue))
            .bmm(pt)
            .bmm(vt.bmm(v)) - v.bmm(p.bmm(v))
        } else {
          val vt = v.t
          val pt = p.t
          val i = STen.eye(avalue.shape(0).toInt, v.options)

          out += v.mm(vt).mm(pt).mm(i - avalue.mm(v)) + (i - v
            .mm(avalue))
            .mm(pt)
            .mm(vt.mm(v)) - v.mm(p.mm(v))
        }
      }
    }
  )
  val value = Variable(
    this,
    implicit scope => implicit forward => a.forward.pinverse(rcond)
  )
}

case class ScatterAdd(
    src: Variable,
    index: STen,
    dim: Int,
    maxIndex: Long
) extends Op {

  val params = List(
    src.zipBackward { case Grad(p, out) =>
      Scope.root { implicit scope => out += p.gather(dim, index) }

    }
  )
  val value =
    Variable(
      this,
      { implicit scope => implicit forward =>
        val srcv = src.forward
        val shape = srcv.sizes.toArray
        assert(shape(dim) == index.shape(dim))
        shape(dim) = maxIndex
        val result = Scope { implicit scope =>
          val zeros = STen.zeros(shape.toList, srcv.options)
          zeros.scatterAdd(dim, index, srcv)
        }
        result
      }
    )

}
case class IndexAdd(
    src: Variable,
    index: STen,
    dim: Int,
    maxIndex: Long
) extends Op {

  val params = List(
    src.zipBackward { case Grad(p, out) =>
      Scope.root { implicit scope =>
        out += p.indexSelect(dim, index)
      }

    }
  )
  val value = {

    Variable(
      this,
      { implicit scope => implicit forward =>
        val srcv = src.forward
        val shape = srcv.sizes.toArray
        shape(dim) = maxIndex
        val result = Scope { implicit scope =>
          val zeros = STen.zeros(shape.toList, srcv.options)
          zeros.indexAdd(dim, index, srcv)
        }
        result
      }
    )
  }
}
case class IndexAddToTarget(
    target: Variable,
    src: Variable,
    index: STen,
    dim: Int
) extends Op {

  val params = List(
    src.zipBackward { case Grad(p, out) =>
      Scope.root { implicit scope =>
        out += p.indexSelect(dim, index)
      }
    },
    target.zipBackward { case Grad(p, out) =>
      out += p
    }
  )
  val value =
    Variable(
      this,
      implicit scope =>
        implicit forward =>
          target.forward
            .indexAdd(dim, index, src.forward)
    )

}
