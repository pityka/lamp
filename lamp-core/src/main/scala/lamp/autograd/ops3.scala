package lamp.autograd

import lamp.STen
import lamp.Scope
import aten.ATen
import aten.Tensor

case class RepeatInterleave(
    self: Variable,
    repeats: STen,
    dim: Int
) extends Op {

  val params = List(
    self.zipBackward { case grad @ Grad(p, out) =>
      implicit val fw = grad.fw
      Scope.root { implicit scope =>
        val plainIndices =
          STen.arange(
            0,
            self.shape.apply(0).toDouble,
            1d,
            self.device.optionsLong
          )
        val repeatedIndices =
          plainIndices.repeatInterleave(repeats, 0)
        val zeros = STen.zerosLike(out)
        val added = zeros.indexAdd(0, repeatedIndices, p)
        out += added
      }

    }
  )

  val value = Variable(
    this,
    implicit scope =>
      implicit forward => self.forward.repeatInterleave(repeats, dim)
  )
}

case class Add(a: Variable, b: Variable) extends Op {

  val params = List(
    a.zipBackward { case grad @ Grad(p, out) =>
      implicit val fw = grad.fw
      Scope.root { implicit scope => 
        out += p.unbroadcast(a.shape) }
    },
    b.zipBackward { case grad @ Grad(p, out) =>
      implicit val fw = grad.fw

      Scope.root { implicit scope => 
        out += p.unbroadcast(b.shape) }

    }
  )
  val value =
    Variable(
      this,
      implicit scope => implicit forward => a.forward.+(b.forward)
    )

}
case class ConstAdd(a: Variable, b: Double) extends Op {

  val params = List(
    a.zipBackward { case grad @ Grad(p, out) =>
      implicit val fw = grad.fw
      Scope.root { implicit scope => out += p.unbroadcast(a.shape) }

    }
  )
  val value =
    Variable(this, implicit scope => implicit forward => a.forward.+(b))

}
case class Minus(a: Variable, b: Variable) extends Op {

  val params = List(
    a.zipBackward { case grad @ Grad(p, out) =>
      implicit val fw = grad.fw
      Scope.root { implicit scope => out += p.unbroadcast(a.shape) }

    },
    b.zipBackward { case grad @ Grad(p, out) =>
      implicit val fw = grad.fw
      Scope.root { implicit scope => out -= p.unbroadcast(b.shape) }

    }
  )
  val value =
    Variable(
      this,
      implicit scope => implicit forward => a.forward.-(b.forward)
    )

}
case class ConstMult(a: Variable, b: Double) extends Op {

  val params = List(
    a.zipBackward { case grad @ Grad(p, out) =>
      implicit val fw = grad.fw
      Scope.root { implicit scope => out += (p * b).unbroadcast(a.shape) }

    }
  )

  val value =
    Variable(this, implicit scope => implicit forward => a.forward.*(b))

}
case class Mult(a: Variable, b: Variable) extends Op {

  val params = List(
    a.zipBackward { case grad @ Grad(p, out) =>
      implicit val fw = grad.fw
      Scope.root { implicit scope =>
        out += (p * b.forward).unbroadcast(a.shape)
      }

    },
    b.zipBackward { case grad @ Grad(p, out) =>
      implicit val fw = grad.fw
      Scope.root { implicit scope =>
        out += (p * a.forward).unbroadcast(b.shape)
      }

    }
  )

  val value = Variable(
    this,
    implicit scope => implicit forward => a.forward.*(b.forward)
  )

}
case class Cross(a: Variable, b: Variable, dim: Int) extends Op {

  val params = List(
    a.zipBackward { case grad @ Grad(p, out) =>
      implicit val fw = grad.fw
      Scope.root { implicit scope =>
        out -= p * (STen.ones(a.shape, p.options).cross(b.forward, dim))

      }

    },
    b.zipBackward { case grad @ Grad(p, out) =>
      implicit val fw = grad.fw
      Scope.root { implicit scope =>
        out += p * (STen.ones(b.shape, p.options).cross(a.forward, dim))
      }

    }
  )

  val value =
    Variable(
      this,
      implicit scope => implicit forward => a.forward.cross(b.forward, dim)
    )

}
case class Div(a: Variable, b: Variable) extends Op {

  val params = List(
    a.zipBackward { case grad @ Grad(p, out) =>
      implicit val fw = grad.fw
      Scope.root { implicit scope =>
        out += (p / b.forward).unbroadcast(a.shape)
      }

    },
    b.zipBackward { case grad =>
      implicit val fw = grad.fw
      Scope.root { implicit scope =>
        val out = grad.out
        val p = grad.p
        val v = value.forward
        val tmp = v / b.forward
        tmp *= p
        val t2 = tmp.unbroadcast(b.shape)
        out -= t2
      }
    }
  )

  val value = Variable(
    this,
    implicit scope => implicit forward => a.forward./(b.forward)
  )
}

case class Sum(a: Variable, dim: List[Int], keepDim: Boolean) extends Op {

  val params = List(a.zipBackward { case Grad(p, out) => out += p })

  val value =
    Variable(
      this,
      implicit scope => implicit forward => a.forward.sum(dim, keepDim)
    )

}

case class Norm2(a: Variable, dim: List[Int], keepDim: Boolean) extends Op {

  val params = List(a.zipBackward { case grad =>
    implicit val fw = grad.fw
    Scope.root { implicit scope =>
      val pa = grad.p * a.forward
      pa /= value.forward
      grad.out += pa
    }
  })

  val value =
    Variable(
      this,
      implicit scope => implicit forward => a.forward.norm2(dim, keepDim)
    )

}

case class ExpandAs(a: Variable, as: STen) extends Op {

  val params = List(a.zipBackward { case grad @ Grad(p, out) =>
    implicit val fw = grad.fw
    Scope.root { implicit scope => out += p.unbroadcast(a.shape) }
  })
  val value =
    Variable(this, implicit scope => implicit forward => a.forward.expandAs(as))
}
case class Expand(a: Variable, shape: List[Long]) extends Op {

  val params = List(a.zipBackward { case grad @ Grad(p, out) =>
    implicit val fw = grad.fw
    Scope.root { implicit scope => out += p.unbroadcast(a.shape) }
  })
  val value =
    Variable(
      this,
      implicit scope => implicit forward => a.forward.expand(shape)
    )
}

// http://cs231n.stanford.edu/handouts/derivatives.pdf
case class MatMul(a: Variable, b: Variable) extends Op {

  val params =
    List(
      a.zipBackward { case grad @ Grad(p, out) =>
        implicit val fw = grad.fw
          Tensor
            .addmm_out_transposed2(
              out.value,
              out.value,
              p.value,
              b.forward.value,
              1d,
              1d
            )
      },
      b.zipBackward { case grad @ Grad(p, out) =>
        implicit val fw = grad.fw
          Tensor
            .addmm_out_transposed1(
              out.value,
              out.value,
              a.forward.value,
              p.value,
              1d,
              1d
            )
      }
    )

  val value = Variable(
    this,
    implicit scope => implicit forward => a.forward.mm(b.forward)
  )

}
case class BatchedMatMul(a: Variable, b: Variable) extends Op {

  val params =
    List(
      a.zipBackward { case grad @ Grad(p, out) =>
        implicit val fw = grad.fw
          Tensor.baddbmm_out_transposed2(
            out.value,
            out.value,
            p.value,
            b.forward.value,
            1d,
            1d
          )
      },
      b.zipBackward { case grad @ Grad(p, out) =>
        implicit val fw = grad.fw
          Tensor.baddbmm_out_transposed1(
            out.value,
            out.value,
            a.forward.value,
            p.value,
            1d,
            1d
          )
      }
    )

  val value = Variable(
    this,
    implicit scope => implicit forward => a.forward.bmm(b.forward)
  )

}
case class EuclideanDistance(a: Variable, b: Variable, dim: Int) extends Op {

  val params =
    List(
      a.zipBackward { case grad =>
        implicit val fw = grad.fw
        Scope.root { implicit scope =>
          val norm = value.forward
          val diff = a.forward.-(b.forward)
          val tmp = diff / norm
          grad.out.addcmulSelf(grad.p, tmp, 1d)
        }

      },
      b.zipBackward { case grad =>
        implicit val fw = grad.fw
        Scope.root { implicit scope =>
          val norm = value.forward
          val diff = a.forward.-(b.forward)
          val tmp = diff / norm
          grad.out.addcmulSelf(grad.p, tmp, -1d)
        }

      }
    )

  val value = Variable(
    this,
    { implicit scope => implicit forward =>
      val diff = a.forward.-(b.forward)

      val norm =
        diff.norm2(List(dim), true)
      norm
    }
  )

}

case class Exp(a: Variable) extends Op {

  val params = List(a.zipBackward { case grad =>
    implicit val fw = grad.fw
      grad.out.addcmulSelf(grad.p, value.forward, 1d)
  })
  val value =
    Variable(this, implicit scope => implicit forward => a.forward.exp)
}
case class CappedShiftedNegativeExponential(
    scope: Scope,
    a: Variable,
    shift: Double
) extends Op {

  val params = List(
    a.zipBackward { case grad @ Grad(p, out) =>
      implicit val fw = grad.fw
      Scope.root { implicit scope =>
        val zeros = STen.zeros(List(1), out.options)
        val result = value.forward.value
        val nonzeros = STen.owned(ATen.mul_1(result, -1d))
        val avv = a.forward.value
        val pred = scope.apply(ATen.le_0(avv, shift))
        val tmp = STen.where(pred, zeros, nonzeros)
        out.addcmulSelf(p, tmp, 1d)
      }
    }
  )

  val value = Variable(
    this,
    { implicit scope => implicit forward =>
      val avv = a.forward.value
      val aOpt = avv.options
      val pred = scope.apply(ATen.le_0(avv, shift))
      val ones =
        scope.apply(ATen.ones(Array(1), aOpt))
      val scalar = scope.apply(ATen.scalar_tensor(shift, aOpt))
      val above = scope.apply(ATen.sub_0(scalar, avv, 1d))
      ATen.exp_(above)
      val result = ATen.where_0(pred, ones, above)
      STen.owned(result)
    }
  )
}
case class LogDet(a: Variable) extends Op {

  val params = List(a.zipBackward { case grad @ Grad(p, out) =>
    implicit val fw = grad.fw
    Scope.root { implicit scope =>
      val tmp = a.forward.inv.t
      out.addcmulSelf(p, tmp, 1d)
    }
  })
  val value =
    Variable(this, implicit scope => implicit forward => a.forward.det.log)
}
case class Log(a: Variable) extends Op {

  val params = List(a.zipBackward { case grad @ Grad(p, out) =>
    implicit val fw = grad.fw
    Scope.root { implicit scope =>
      val tmp = a.forward.reciprocal
      out.addcmulSelf(p, tmp, 1d)
    }
  })
  val value =
    Variable(this, implicit scope => implicit forward => a.forward.log)
}
case class Log1p(a: Variable) extends Op {

  val params = List(a.zipBackward { case grad @ Grad(p, out) =>
    implicit val fw = grad.fw
    Scope.root { implicit scope =>
      val tmp = a.forward + 1d
      tmp.reciprocal_()
      out.addcmulSelf(p, tmp, 1d)
    }

  })
  val value =
    Variable(this, implicit scope => implicit forward => a.forward.log1p)
}
case class Sin(a: Variable) extends Op {

  val params = List(a.zipBackward { case grad @ Grad(p, out) =>
    implicit val fw = grad.fw
    Scope.root { implicit scope =>
      val tmp = a.forward.cos
      out.addcmulSelf(p, tmp, 1d)
    }

  })
  val value =
    Variable(this, implicit scope => implicit forward => a.forward.sin)
}
case class Cos(a: Variable) extends Op {

  val params = List(a.zipBackward { case grad @ Grad(p, out) =>
    implicit val fw = grad.fw
    Scope.root { implicit scope =>
      val tmp = a.forward.sin
      out.addcmulSelf(p, tmp, -1d)
    }

  })
  val value =
    Variable(this, implicit scope => implicit forward => a.forward.cos)
}
case class Tan(a: Variable) extends Op {

  val params = List(a.zipBackward { case grad @ Grad(p, out) =>
    implicit val fw = grad.fw
    Scope.root { implicit scope =>
      val tmp1 = value.forward.pow(2d)
      val one = STen.ones(List(1), tmp1.options)
      tmp1 += one
      out.addcmulSelf(p, tmp1, 1d)
    }

  })
  val value =
    Variable(this, implicit scope => implicit forward => a.forward.tan)
}
case class Tanh(a: Variable) extends Op {

  val params = List(a.zipBackward { case grad @ Grad(p, out) =>
    implicit val fw = grad.fw
    Scope.root { implicit scope =>
      val tmp1 = STen.tanh_backward(p, value.forward)
      out += tmp1
    }
  })
  val value =
    Variable(this, implicit scope => implicit forward => a.forward.tanh)
}
case class ArcTan(a: Variable) extends Op {

  val params = List(a.zipBackward { case grad @ Grad(p, out) =>
    implicit val fw = grad.fw
    Scope.root { implicit scope =>
      val tmp1 = a.forward.pow(2d)
      val one = STen.ones(List(1L), tmp1.options)
      tmp1 += one
      tmp1.reciprocal_()
      out.addcmulSelf(p, tmp1, 1d)

    }

  })
  val value =
    Variable(this, implicit scope => implicit forward => a.forward.atan)
}
case class PowConst(a: Variable, exponent: Double) extends Op {

  val params = List(a.zipBackward { case grad @ Grad(p, out) =>
    implicit val fw = grad.fw
    Scope.root { implicit scope =>
      val tmp1 = a.forward.pow(exponent - 1)
      out.addcmulSelf(p, tmp1, exponent)
    }

  })
  val value =
    Variable(
      this,
      implicit scope => implicit forward => a.forward.pow(exponent)
    )
}
case class Pow(a: Variable, exponent: Variable) extends Op {

  val params = List(
    a.zipBackward { case grad @ Grad(p, out) =>
      implicit val fw = grad.fw
      Scope.root { implicit scope =>
        val exp = exponent.forward.toDoubleArray.apply(0)
        val tmp1 = a.forward.pow(exp - 1)
        out.addcmulSelf(p, tmp1, exp)
      }

    },
    exponent.zipBackward { case grad @ Grad(p, out) =>
      implicit val fw = grad.fw
      Scope.root { implicit scope =>
        val exp = exponent.forward.toDoubleArray.apply(0)
        val av = a.forward
        val tmp1 = av.pow(exp)
        val tmp2 = av.log
        val tmp3 = tmp1 * tmp2
        val p2 = p.unbroadcast(
          List(if (out.shape.isEmpty) 1 else out.shape.toList.head, 1)
        )
        val tmp4 = tmp3.sum
        out.addcmulSelf(p2, tmp4, 1d)
      }
    }
  )
  val value =
    Variable(
      this,
      implicit scope => implicit forward => a.forward.pow(exponent.forward)
    )
}
case class Relu(a: Variable) extends Op {

  val params = List(
    a.zipBackward { case grad @ Grad(p, out) =>
      implicit val fw = grad.fw
      Scope.root { implicit scope =>
        val av = a.forward
        val aOpt = av.options
        val pred = av.lt(0d)
        val ones =
          STen.ones(List(1), aOpt)
        val zeros =
          STen.zeros(List(1), aOpt)
        val tmp = STen.where(pred, zeros, ones)
        out.addcmulSelf(p, tmp, 1d)
      }
    }
  )

  val value =
    Variable(this, implicit scope => implicit forward => a.forward.relu)
}
case class LeakyRelu(a: Variable, slope: Double) extends Op {

  val params = List(
    a.zipBackward { case grad @ Grad(p, out) =>
      implicit val fw = grad.fw
      Scope.root { implicit scope =>
        val av = a.forward
        val aOpt = av.options
        val pred = av.lt(0d)
        val ones =
          STen.ones(List(1), aOpt)
        val s =
          STen.zeros(List(1), aOpt) + slope
        val tmp = STen.where(pred, s, ones)
        out.addcmulSelf(p, tmp, 1d)
      }
    }
  )
  val value =
    Variable(
      this,
      implicit scope => implicit forward => a.forward.leakyRelu(slope)
    )
}

case class LogSoftMax(a: Variable, dim: Int) extends Op {

  val params = List(
    a.zipBackward { case grad @ Grad(p, out) =>
      implicit val fw = grad.fw
      Scope.root { implicit scope =>
        val v = value.forward.value
        val tmp = STen.owned(
          ATen._log_softmax_backward_data(
            p.value,
            v,
            dim,
            a.scalarTypeByte
          )
        )
        out += tmp

      }
    }
  )
  val value =
    Variable(
      this,
      implicit scope => implicit forward => a.forward.logSoftMax(dim)
    )

}
case class Gelu(a: Variable) extends Op {

  val params = List(
    a.zipBackward { case grad @ Grad(p, out) =>
      implicit val fw = grad.fw
      Scope.root { implicit scope =>
        val tmp = STen.owned(ATen.gelu_backward(p.value, a.forward.value))
        out += tmp
      }
    }
  )
  val value =
    Variable(this, implicit scope => implicit forward => a.forward.gelu)

}
case class Softplus(a: Variable, beta: Double, threshold: Double) extends Op {

  val params = List(
    a.zipBackward { case grad @ Grad(p, out) =>
      implicit val fw = grad.fw
      Scope.root { implicit scope =>
        val tmp =
          STen.softplus_backward(p, a.forward, beta, threshold)
        out += tmp
      }
    }
  )
  val value = Variable(
    this,
    implicit scope => implicit forward => a.forward.softplus(beta, threshold)
  )

}
case class Sigmoid(a: Variable) extends Op {

  val params = List(
    a.zipBackward { case grad @ Grad(p, out) =>
      implicit val fw = grad.fw
      Scope.root { implicit scope =>
        val tmp =
          STen.owned(
            ATen.sigmoid_backward(p.value, value.forward.value)
          )
        out += tmp
      }

    }
  )
  val value =
    Variable(this, implicit scope => implicit forward => a.forward.sigmoid)

}
case class HardSwish(a: Variable) extends Op {

  val params = List(
    a.zipBackward { case grad @ Grad(p, out) =>
      implicit val fw = grad.fw
      Scope.root { implicit scope =>
        val tmp =
          STen.owned(ATen.hardswish_backward(p.value, a.forward.value))
        out += tmp
      }

    }
  )
  val value =
    Variable(this, implicit scope => implicit forward => a.forward.hardSwish)

}
