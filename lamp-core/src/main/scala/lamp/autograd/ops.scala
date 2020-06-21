package lamp.autograd
import org.saddle._
import java.{util => ju}
import aten.Tensor
import aten.ATen
import aten.TensorOptions
import TensorHelpers.{unbroadcast => ub}
import lamp.syntax
import lamp.util.NDArray
import lamp.FloatingPointPrecision
import lamp.DoublePrecision
import lamp.SinglePrecision

case class Constant(const: Tensor) extends Op {
  val params = Nil
  val value = Variable(this, const, leaf = true)
}
case class Transpose(a: Variable) extends Op {
  val params = List(
    a.zipBackward { (p, out) =>
      val tmp = p.transpose(0L, 1L)
      ATen.add_out(out, out, tmp, 1d)
      tmp.release()
    }
  )
  val value = Variable(this, a.value.transpose(0L, 1L))
}
case class View(a: Variable, shape: Array[Long]) extends Op {
  val params = List(
    a.zipBackward { (p, out) =>
      val selected = ATen._unsafe_view(p, out.sizes)
      ATen.add_out(out, out, selected, 1d)
      selected.release
    }
  )
  val value = Variable(this, ATen._unsafe_view(a.value, shape))
}

/** Prepends a new dimension and concatenates the tensors along that axis
  */
case class ConcatenateAddNewDim(a: Seq[Variable]) extends Op {
  val params = a.zipWithIndex.toList.map {
    case (a, idx) =>
      a.zipBackward { (p, out) =>
        val selected = ATen.select(p, 0, idx.toLong)
        ATen.add_out(out, out, selected, 1d)
        selected.release
      }
  }
  val shapes = a.map(_.shape)

  assert(
    shapes.distinct.size == 1,
    "Concatenable tensors must have the same shape."
  )
  val shape = List(a.size.toLong) ++ shapes.head
  val shape1 = List(1L) ++ shapes.head
  val zeros = ATen.zeros(shape.toArray, a.head.options)
  a.foldLeft(0) { (offset, v) =>
    val viewed = ATen._unsafe_view(v.value, shape1.toArray)
    val idx = Tensor.scalarLong(offset, a.head.options.toLong)
    ATen._index_copy_(zeros, 0, idx, viewed)
    idx.release
    viewed.release
    offset + 1
  }
  val value = Variable(this, zeros)
}
case class Select(a: Variable, dim: Long, index: Long) extends Op {
  val params = List(
    a.zipBackward { (p, out) =>
      val tmp = ATen.zeros(out.sizes, out.options())
      val scalar = Tensor.scalarLong(index, out.options().toLong())
      val pshape = p.shape
      val reshape = pshape.take(dim.toInt) ::: (1L :: pshape.drop(dim.toInt))
      val p2 = ATen._unsafe_view(p, reshape.toArray)
      val tmp2 = ATen
        .index_add(tmp, dim, scalar, p2)
      ATen.add_out(out, out, tmp2, 1d)
      tmp.release()
      scalar.release
      p2.release
      tmp2.release()
    }
  )
  val value = Variable(this, ATen.select(a.value, dim, index))
}
case class ArgMax(a: Variable, dim: Long, keepDim: Boolean) extends Op {
  val params = List(
    a.zipBackward { (p, out) =>
      throw new RuntimeException("Argmax is not differentiable")
    }
  )
  val value = Variable(this, ATen.argmax(a.value, dim, keepDim))
}
case class Assign(a: Variable, b: Variable) extends Op {
  val params = List(
    a.zipBackward { (p, out) =>
      throw new RuntimeException("Argmax is not differentiable")
    }
  )
  val value = Variable(this, b.value)
}
case class OneHot(a: Variable, numClasses: Int) extends Op {
  val params = List(
    a.zipBackward { (p, out) =>
      throw new RuntimeException("OneHot is not differentiable")
    }
  )
  val value = Variable(this, ATen.one_hot(a.value, numClasses))
}
case class CastToPrecision(a: Variable, precision: FloatingPointPrecision)
    extends Op {
  val params = List(
    a.zipBackward { (p, out) =>
      throw new RuntimeException("Cast to Precision is not differentiable")
    }
  )
  val value = Variable(this, {
    precision match {
      case DoublePrecision => ATen._cast_Double(a.value, false)
      case SinglePrecision => ATen._cast_Float(a.value, false)
    }
  })
}

case class Add(a: Variable, b: Variable) extends Op {
  val params = List(
    a.zipBackward { (p, out) =>
      val p2 = ub(p, a.sizes)
      ATen.add_out(out, out, p2, 1d)
      p2.release()
    },
    b.zipBackward { (p, out) =>
      val p2 = ub(p, b.sizes)
      ATen.add_out(out, out, p2, 1d)
      p2.release()
    }
  )
  val value = Variable(this, ATen.add_0(a.value, b.value, 1d))

}
case class ConstAdd(a: Variable, b: Double) extends Op {
  val params = List(
    a.zipBackward { (p, out) =>
      val p2 = ub(p, a.sizes)
      ATen.add_out(out, out, p2, 1d)
      p2.release()
    }
  )
  val value = Variable(this, ATen.add_1(a.value, b, 1d))

}
case class Minus(a: Variable, b: Variable) extends Op {
  val params = List(
    a.zipBackward { (p, out) =>
      val p2 = ub(p, a.sizes)
      ATen.add_out(out, out, p2, 1d)
      p2.release()
    },
    b.zipBackward { (p, out) =>
      val p2 = ub(p, b.sizes)
      ATen.add_out(out, out, p2, -1d)
      p2.release()
    }
  )
  val value = Variable(this, ATen.add_0(a.value, b.value, -1d))

}
case class ConstMult(a: Variable, b: Double) extends Op {
  val params = List(
    a.zipBackward { (p, out) =>
      val tmp = ATen.mul_1(p, b)
      val tmp2 = ub(tmp, a.sizes)
      ATen.add_out(out, out, tmp2, 1d)
      tmp.release
      tmp2.release
    }
  )

  val value = Variable(this, ATen.mul_1(a.value, b))

}
case class Mult(a: Variable, b: Variable) extends Op {
  val params = List(
    a.zipBackward { (p, out) =>
      val tmp = ATen.mul_0(p, b.value)
      val tmp2 = ub(tmp, a.sizes)
      ATen.add_out(out, out, tmp2, 1d)
      tmp2.release
      tmp.release
    },
    b.zipBackward { (p, out) =>
      val tmp = ATen.mul_0(p, a.value)
      val tmp2 = ub(tmp, b.sizes)
      ATen.add_out(out, out, tmp2, 1d)
      tmp2.release
      tmp.release
    }
  )

  val value = Variable(this, ATen.mul_0(a.value, b.value))

}
case class Div(a: Variable, b: Variable) extends Op {
  val params = List(
    a.zipBackward { (p, out) =>
      val tmp = ATen.div_0(p, b.value)
      val t2 = ub(tmp, a.sizes)
      ATen.add_out(out, out, t2, 1d)
      tmp.release
      t2.release
    },
    b.zipBackward { (p, out) =>
      // out += p * (a.value * b.value.map(x => -1d / (x * x)))
      val tmp = ATen.div_0(value.value, b.value)
      ATen.mul_out(tmp, tmp, p)
      val t2 = ub(tmp, b.sizes)
      ATen.add_out(out, out, t2, -1d)
      tmp.release()
      t2.release()
    }
  )

  val value = Variable(this, ATen.div_0(a.value, b.value))

}

case class Sum(a: Variable) extends Op {
  val params = List(a.zipBackward { (p, out) => ATen.add_out(out, out, p, 1d) })

  val value = Variable(this, ATen.sum_0(a.value))

}
case class ColSum(a: Variable) extends Op {
  val params = List(a.zipBackward { (p, out) => ATen.add_out(out, out, p, 1d) })

  val value = Variable(this, ATen.sum_1(a.value, Array(1), true))

}
case class RowSum(a: Variable) extends Op {
  val params = List(a.zipBackward { (p, out) => ATen.add_out(out, out, p, 1d) })

  val value = Variable(this, ATen.sum_1(a.value, Array(0), true))

}

// http://cs231n.stanford.edu/handouts/derivatives.pdf
case class MatMul(a: Variable, b: Variable) extends Op {
  val params =
    List(
      a.zipBackward { (p, out) =>
        val bt = b.value.transpose(0L, 1L)
        ATen.addmm_out(out, out, p, bt, 1d, 1d)
        bt.release
      },
      b.zipBackward { (p, out) =>
        val at = a.value.transpose(0L, 1L)
        ATen.addmm_out(out, out, at, p, 1d, 1d)
        at.release
      }
    )

  val value = Variable(this, ATen.mm(a.value, b.value))

}

case class Exp(a: Variable) extends Op {
  val params = List(
    a.zipBackward { (p, out) => ATen.addcmul_out(out, out, p, value.value, 1d) }
  )
  val value = Variable(this, ATen.exp(a.value))
}
case class Log(a: Variable) extends Op {
  val params = List(a.zipBackward { (p, out) =>
    val tmp = ATen.reciprocal(a.value)
    ATen.addcmul_out(out, out, p, tmp, 1d)
    tmp.release
  })
  val value = Variable(this, ATen.log(a.value))
}
case class Sin(a: Variable) extends Op {
  val params = List(a.zipBackward { (p, out) =>
    val tmp = ATen.cos(a.value)
    ATen.addcmul_out(out, out, p, tmp, 1d)
    tmp.release
  })
  val value = Variable(this, ATen.sin(a.value))
}
case class Cos(a: Variable) extends Op {
  val params = List(a.zipBackward { (p, out) =>
    val tmp = ATen.sin(a.value)
    ATen.addcmul_out(out, out, p, tmp, -1d)
    tmp.release
  })
  val value = Variable(this, ATen.cos(a.value))
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
}
case class Tanh(a: Variable) extends Op {
  val params = List(a.zipBackward { (p, out) =>
    val tmp1 = ATen.tanh_backward(p, value.value)
    ATen.add_out(out, out, tmp1, 1d)
    tmp1.release
  })
  val value = Variable(this, ATen.tanh(a.value))
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
}
case class PowConst(a: Variable, exponent: Double) extends Op {
  val params = List(a.zipBackward { (p, out) =>
    val tmp1 = ATen.pow_0(a.value, exponent - 1)
    ATen.addcmul_out(out, out, p, tmp1, exponent)
    tmp1.release
  })
  val value = Variable(this, ATen.pow_0(a.value, exponent))
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
      pred.release
      tmp.release
      ones.release
      zeros.release
    }
  )
  val value = Variable(this, ATen.relu(a.value))
}

case class LogSoftMax(a: Variable, dim: Int) extends Op {

  val params = List(
    a.zipBackward { (p, out) =>
      val tmp = ATen._log_softmax_backward_data(p, value.value, dim, a.value)
      ATen.add_out(out, out, tmp, 1d)
      tmp.release
    }
  )
  val value = Variable(this, ATen.log_softmax(a.value, dim))

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

}
case class Sigmoid(a: Variable) extends Op {

  val params = List(
    a.zipBackward { (p, out) =>
      val tmp = ATen.sigmoid_backward(p, value.value)
      ATen.add_out(out, out, tmp, 1d)
      tmp.release
    }
  )
  val value = Variable(this, ATen.sigmoid(a.value))

}

case class Mean(a: Variable, dim: List[Int]) extends Op {

  val params = List(
    a.zipBackward { (p, out) =>
      ATen.add_out(
        out,
        out,
        p,
        1d / dim.map(l => a.sizes.apply(l.toInt)).sum
      )
    }
  )
  val value =
    Variable(this, ATen.mean_1(a.value, dim.map(_.toLong).toArray, true))

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
  val value =
    Variable(this, ATen.mul_0(a.value, mask), releaseWith = List(mask))

}

// https://arxiv.org/pdf/1602.07868.pdf
case class WeightNorm(v: Variable, g: Variable, dim: Long) extends Op {
  assert(v.sizes.size == 2, "WeightNorm: v should have 2 dimensions")
  assert(
    g.sizes.toList == List(1, v.sizes(1)),
    "WeightNorm: g should have dimensions 1 x a where a is the second dimension of v."
  )
  def gradg(p: Tensor) = {
    val tmp0 = ATen.mul_0(p, v.value)
    val tmp1 = ATen.sum_1(tmp0, Array(0), false)
    ATen.div_out(tmp1, tmp1, norm)
    tmp0.release
    tmp1
  }

  // https://arxiv.org/pdf/1602.07868.pdf eq3
  // Mind the dot product (.)
  val params = List(
    v.zipBackward { (p, out) =>
      val tmp1 = ATen.div_0(g.value, norm)
      val tmp3 = ATen.mul_0(tmp1, p)
      val gg = gradg(p)
      val tmp2 = ATen.mul_0(g.value, gg)
      ATen.div_out(tmp2, tmp2, norm)
      ATen.div_out(tmp2, tmp2, norm)
      val tmp4 = ATen.mul_0(tmp2, v.value)
      ATen.add_out(tmp3, tmp3, tmp4, -1d)
      ATen.add_out(out, out, tmp3, 1d)
      tmp1.release
      tmp2.release
      tmp3.release
      tmp4.release
      gg.release
    },
    g.zipBackward { (p, out) =>
      val tmp2 = gradg(p)
      ATen.add_out(out, out, tmp2, 1d)

    }
  )
  // https://arxiv.org/pdf/1602.07868.pdf eq2
  val norm =
    ATen.norm_2(v.value, Array(dim), false, v.options.scalarTypeByte)
  val w = ATen.mul_0(v.value, g.value)
  ATen.div_out(w, w, norm)

  val value = Variable(this, w, releaseWith = List(norm))

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
    weights: Tensor,
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
      value1,
      releaseWith = List(total_weight)
    )

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
}

/** 1D convolution
  *
  * @param input batch x in_channels x L
  * @param weight out_channels x in_channels x kernel_size
  * @param bias out_channels
  * @return Variable with Tensor of size batch x out_channels x L' (length depends on stride/padding/dilation)
  */
case class Conv1D(
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

  override val params: List[(Variable, (Tensor, Tensor) => Unit)] = List(
    weight.zipBackward { (p, out) =>
      val pSize = p.sizes

      val p_repeated = ATen.repeat_interleave_2(p, inputChannels / groups, 1)
      val p_repeated_size = p_repeated.sizes
      val p_repeated_viewed =
        ATen._unsafe_view(
          p_repeated,
          Array(p_repeated_size(0) * p_repeated_size(1), 1, p_repeated_size(2))
        )
      val input_viewed = ATen._unsafe_view(
        input.value,
        Array(1, batchSize * inputChannels, imageSize)
      )
      val zero = ATen.zeros(Array(p_repeated_viewed.sizes.apply(0)), p.options)
      val conv_0 = ATen.conv1d(
        input_viewed,
        p_repeated_viewed,
        zero,
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
      val conv_1_sum_viewed_transposed = conv_1_sum_viewed.transpose(0, 1)

      val conv_1_sum_viewed_transposed_narrowed =
        ATen.narrow_0(conv_1_sum_viewed_transposed, 2, 0, kernelSize)
      ATen.add_out(out, out, conv_1_sum_viewed_transposed_narrowed, 1d)

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
    input.zipBackward { (p, out) =>
      val pSize = p.sizes
      val zeros = ATen.zeros(Array(inputChannels), p.options)
      val outputSizeWithoutExtraPadding =
        (pSize(2) - 1) * stride - 2 * padding + dilation * (kernelSize - 1) + 1
      val extraPadding = out.sizes.apply(2) - outputSizeWithoutExtraPadding
      val tmp = ATen.conv_transpose1d(
        p,
        weight.value,
        zeros,
        Array(stride),
        Array(padding),
        Array(extraPadding),
        groups,
        Array(dilation)
      )
      ATen.add_out(out, out, tmp, 1d)
      tmp.release
      zeros.release()
    },
    bias.zipBackward { (p, out) =>
      val p2 = ub(p, List(out.sizes.toList.head, 1))
      val p3 = ATen._unsafe_view(p2, out.sizes)
      ATen.add_out(out, out, p3, 1d)
      p2.release()
      p3.release()
    }
  )

  val value =
    Variable(this, {
      ATen.conv1d(
        input.value,
        weight.value,
        bias.value,
        Array(stride),
        Array(padding),
        Array(dilation),
        groups
      )
    })
}

/** 2D convolution
  *
  * @param input batch x in_channels x height x width
  * @param weight out_channels x in_channels x kernel_size x kernel_size
  * @param bias out_channels
  * @return Variable with Tensor of size batch x out_channels x L' (length depends on stride/padding/dilation)
  */
case class Conv2D(
    input: Variable,
    weight: Variable,
    bias: Variable,
    stride: Long,
    padding: Long,
    dilation: Long,
    groups: Long
) extends Op {

  assert(input.shape.size == 4, "Input dimensions must be 3")
  assert(weight.shape.size == 4, "Weight dimensions must be 3")
  val batchSize = input.shape(0)
  val inputChannels = input.shape(1)
  val imageHeight = input.shape(2)
  val imageWidth = input.shape(3)
  val kernelSize = weight.shape(2)
  val outChannels = weight.shape(0)
  assert(
    weight.shape(1) == inputChannels,
    s"Weight 2nd dimension must have size equal to input channels (2nd dim of input). Got weight: ${weight.shape}, input: ${input.shape} "
  )
  assert(
    bias.shape(0) == outChannels,
    "Number of biases must be the number of output channels"
  )

  override val params: List[(Variable, (Tensor, Tensor) => Unit)] = List(
    weight.zipBackward { (p, out) =>
      if (p.options().isCuda()) {
        val tmp = ATen.cudnn_convolution_backward_weight(
          weight.shape.toArray,
          p,
          input.value,
          Array(padding, padding),
          Array(stride, stride),
          Array(dilation, dilation),
          groups,
          false,
          false
        )
        ATen.add_out(out, out, tmp, 1d)
        tmp.release
      } else {
        val pSize = p.sizes

        val p_repeated = ATen.repeat_interleave_2(p, inputChannels / groups, 1)
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
          input.value,
          Array(1, batchSize * inputChannels, imageHeight, imageWidth)
        )
        val zero =
          ATen.zeros(Array(p_repeated_viewed.sizes.apply(0)), p.options)
        val conv_0 = ATen.conv2d(
          input_viewed,
          p_repeated_viewed,
          zero,
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
        val conv_1_sum_viewed_transposed = conv_1_sum_viewed.transpose(0, 1)

        val conv_1_sum_viewed_transposed_narrowed1 =
          ATen.narrow_0(conv_1_sum_viewed_transposed, 2, 0, kernelSize)
        val conv_1_sum_viewed_transposed_narrowed2 =
          ATen
            .narrow_0(conv_1_sum_viewed_transposed_narrowed1, 3, 0, kernelSize)

        ATen.add_out(out, out, conv_1_sum_viewed_transposed_narrowed2, 1d)

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
    },
    input.zipBackward { (p, out) =>
      if (p.options().isCuda()) {
        val tmp = ATen.cudnn_convolution_backward_input(
          input.shape.toArray,
          p,
          weight.value,
          Array(padding, padding),
          Array(stride, stride),
          Array(dilation, dilation),
          groups,
          false,
          false
        )
        ATen.add_out(out, out, tmp, 1d)
        tmp.release
      } else {
        val pSize = p.sizes
        val zeros = ATen.zeros(Array(inputChannels), p.options)
        val outputSizeWithoutExtraPadding =
          (pSize(2) - 1) * stride - 2 * padding + dilation * (kernelSize - 1) + 1
        val extraPaddingH = out.sizes.apply(2) - outputSizeWithoutExtraPadding
        val extraPaddingW = out.sizes.apply(3) - outputSizeWithoutExtraPadding
        val tmp = ATen.conv_transpose2d(
          p,
          weight.value,
          zeros,
          Array(stride),
          Array(padding),
          Array(extraPaddingH, extraPaddingW),
          groups,
          Array(dilation)
        )
        ATen.add_out(out, out, tmp, 1d)
        tmp.release
        zeros.release()
      }
    },
    bias.zipBackward { (p, out) =>
      val p2 = ub(p, List(out.sizes.toList.head, 1, 1))
      val p3 = ATen._unsafe_view(p2, out.sizes)

      ATen.add_out(out, out, p3, 1d)
      p2.release()
      p3.release()
    }
  )

  val value =
    Variable(this, {
      ATen.conv2d(
        input.value,
        weight.value,
        bias.value,
        Array(stride),
        Array(padding),
        Array(dilation),
        groups
      )
    })
}

/** 1D max pooling
  *
  * @param input batch x in_channels x L
  */
case class MaxPool1D(
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

  override val params: List[(Variable, (Tensor, Tensor) => Unit)] = List(
    input.zipBackward { (p, out) =>
      val zeros = ATen.zeros_like(out, out.options())
      val p_flatten = ATen.flatten(p, 0, 1)
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
      val catted_viewed = ATen._unsafe_view(catted, out.sizes)
      ATen.add_out(out, out, catted_viewed, 1d)

      catted_viewed.release
      catted.release
      zeros_flatten.release
      mask_flatten.release
      p_flatten.release
      zeros.release
    }
  )

  val (output, mask) = ATen.max_pool1d_with_indices(
    input.value,
    Array(kernelSize),
    Array(stride),
    Array(padding),
    Array(dilation),
    false
  )

  val value =
    Variable(this, output, releaseWith = List(mask))
}

/** 2D max pooling
  *
  * @param input batch x in_channels x h x w
  */
case class MaxPool2D(
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

  override val params: List[(Variable, (Tensor, Tensor) => Unit)] = List(
    input.zipBackward { (p, out) =>
      val tmp = ATen.max_pool2d_with_indices_backward(
        p,
        input.value,
        Array(kernelSize),
        Array(stride),
        Array(padding),
        Array(dilation),
        false,
        mask
      )

      ATen.add_out(out, out, tmp, 1d)
      tmp.release

    }
  )

  val (output, mask) = ATen.max_pool2d_with_indices(
    input.value,
    Array(kernelSize),
    Array(stride),
    Array(padding),
    Array(dilation),
    false
  )

  val value =
    Variable(this, output, releaseWith = List(mask))
}

/** 2D avg pooling
  *
  * @param input batch x in_channels x h x w
  */
case class AvgPool2D(
    input: Variable,
    kernelSize: Long,
    stride: Long,
    padding: Long
) extends Op {

  assert(input.shape.size == 4, "Input dimensions must be 4")
  val batchSize = input.shape(0)
  val inputChannels = input.shape(1)
  val imageSize = input.shape(2)

  override val params: List[(Variable, (Tensor, Tensor) => Unit)] = List(
    input.zipBackward { (p, out) =>
      val tmp = ATen.avg_pool2d_backward(
        p,
        input.value,
        Array(kernelSize),
        Array(stride),
        Array(padding),
        false,
        true,
        Long.MinValue
      )

      ATen.add_out(out, out, tmp, 1d)
      tmp.release

    }
  )

  val value =
    Variable(
      this,
      ATen.avg_pool2d(
        input.value,
        Array(kernelSize),
        Array(stride),
        Array(padding),
        false,
        true,
        Long.MinValue
      )
    )
}

case class FlattenLastDimensions(
    input: Variable,
    last: Int
) extends Op {

  assert(input.shape.size >= 2, "Input dimensions must be 4")

  override val params: List[(Variable, (Tensor, Tensor) => Unit)] = List(
    input.zipBackward { (p, out) =>
      val tmp = ATen._unsafe_view(p, out.shape.toArray)
      ATen.add_out(out, out, tmp, 1d)
      tmp.release

    }
  )

  val size = input.shape.takeRight(last).reduce(_ * _)

  val value =
    Variable(
      this,
      ATen._unsafe_view(input.value, Array(-1, size))
    )
}

/* 0-th dimension has samples. Everything else is flattened out into features. */
case class BatchNorm(
    input: Variable,
    weight: Variable,
    bias: Variable,
    runningMean: Tensor,
    runningVar: Tensor,
    training: Boolean,
    momentum: Double,
    eps: Double
) extends Op {
  val input_flattened = ATen.flatten(input.value, 1, input.shape.size - 1)
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
    weight.value,
    bias.value,
    runningMean,
    runningVar,
    training,
    momentum,
    eps
  )
  val output_reshaped = ATen._unsafe_view(output, input.shape.toArray)

  override val value: Variable =
    Variable(
      this,
      output_reshaped,
      releaseWith = List(saveMean, saveInvstd, input_flattened, output)
    )

  override val params: List[(Variable, (Tensor, Tensor) => Unit)] = List(
    input.zipBackward { (p, out) =>
      val flattened_p =
        ATen.flatten(p, 1, p.shape.size - 1)
      val (gradInput, a, b) = ATen.native_batch_norm_backward(
        flattened_p,
        input_flattened,
        weight.value,
        runningMean,
        runningVar,
        saveMean,
        saveInvstd,
        training,
        eps,
        Array(true, true, true)
      )
      val gradInput_reshaped = ATen._unsafe_view(gradInput, out.shape.toArray)
      ATen.add_out(out, out, gradInput_reshaped, 1d)
      gradInput.release
      a.release
      b.release
      flattened_p.release()
      gradInput_reshaped.release
    },
    weight.zipBackward { (p, out) =>
      val flattened_p =
        ATen.flatten(p, 1, p.shape.size - 1)
      val (a, gradWeight, b) = ATen.native_batch_norm_backward(
        flattened_p,
        input_flattened,
        weight.value,
        runningMean,
        runningVar,
        saveMean,
        saveInvstd,
        training,
        eps,
        Array(true, true, true)
      )
      val grad_reshaped = ATen._unsafe_view(gradWeight, out.shape.toArray)
      ATen.add_out(out, out, grad_reshaped, 1d)
      gradWeight.release
      grad_reshaped.release
      a.release
      b.release
      flattened_p.release()
    },
    bias.zipBackward { (p, out) =>
      val flattened_p =
        ATen.flatten(p, 1, p.shape.size - 1)
      val tmp = ub(flattened_p, out.shape)
      ATen.add_out(out, out, tmp, 1d)
      tmp.release
      flattened_p.release
    }
  )
}

/** Batch Norm 2D
  * 0-th dimension are samples. 1-th are features, everything else is averaged out.
  */
case class BatchNorm2D(
    input: Variable,
    weight: Variable,
    bias: Variable,
    runningMean: Tensor,
    runningVar: Tensor,
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
    input.value,
    weight.value,
    bias.value,
    runningMean,
    runningVar,
    training,
    momentum,
    eps
  )
  override val value: Variable =
    Variable(
      this,
      output,
      releaseWith = List(saveMean, saveInvstd)
    )

  override val params: List[(Variable, (Tensor, Tensor) => Unit)] = List(
    input.zipBackward { (p, out) =>
      val (gradInput, a, b) = ATen.native_batch_norm_backward(
        p,
        input.value,
        weight.value,
        runningMean,
        runningVar,
        saveMean,
        saveInvstd,
        training,
        eps,
        Array(true, true, true)
      )
      val gradInput_reshaped = ATen._unsafe_view(gradInput, out.shape.toArray)
      ATen.add_out(out, out, gradInput_reshaped, 1d)
      gradInput.release
      a.release
      b.release
      gradInput_reshaped.release
    },
    weight.zipBackward { (p, out) =>
      val (a, gradWeight, b) = ATen.native_batch_norm_backward(
        p,
        input.value,
        weight.value,
        runningMean,
        runningVar,
        saveMean,
        saveInvstd,
        training,
        eps,
        Array(true, true, true)
      )
      val grad_reshaped = ATen._unsafe_view(gradWeight, out.shape.toArray)
      ATen.add_out(out, out, grad_reshaped, 1d)
      gradWeight.release
      grad_reshaped.release
      a.release
      b.release
    },
    bias.zipBackward { (p, out) =>
      val tmp = ub(p, (out.shape ++ (1 to p.shape.size - 2).map(_ => 1L)))
      val tmp_viewed = ATen._unsafe_view(
        tmp,
        out.shape.toArray
      )
      ATen.add_out(out, out, tmp_viewed, 1d)
      tmp.release
      tmp_viewed.release
    }
  )
}
case class Embedding(
    input: Variable,
    weight: Variable
) extends Op {

  assert(input.shape.size >= 1, "Input dimensions must be at least 1")
  assert(weight.shape.size == 2, "Weight must have 2 dimensions")

  override val params: List[(Variable, (Tensor, Tensor) => Unit)] = List(
    weight.zipBackward { (p, out) =>
      val tmp = ATen
        .embedding_backward(p, input.value, weight.shape(0), 0L, false, false)
      ATen.add_out(out, out, tmp, 1d)
      tmp.release

    }
  )

  val value =
    Variable(
      this,
      ATen.embedding(weight.value, input.value, 0L, false, false)
    )
}
