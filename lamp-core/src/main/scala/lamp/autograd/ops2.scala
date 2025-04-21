package lamp.autograd

import lamp.STen
import lamp.Scope
import aten.ATen

case class Mean(a: Variable, dim: List[Int], keepDim: Boolean) extends Op {

  val params = List(
    a.zipBackward { case grad @ Grad(p, out) =>
      implicit val fw = grad.fw
      STen.addOut(
        out,
        out,
        p,
        1d / dim.map(l => a.shape.apply(l.toInt)).foldLeft(1L)(_ * _)
      )

    }
  )
  val value =
    Variable(
      this,
      implicit scope => implicit forward => a.forward.mean(dim, keepDim)
    )

}
case class Variance(a: Variable, dim: List[Int]) extends Op {

  val params = List(
    a.zipBackward { case grad @ Grad(p, out) =>
      implicit val fw = grad.fw
      Scope.root { implicit scope =>
        val av = a.forward
        val m = value.forwardAux._2.head
        val s = av - m
        out.addcmulSelf(
          p,
          s,
          2d / (dim.map(l => a.shape.apply(l.toInt)).sum - 1)
        )

      }

    }
  )

  val value =
    Variable.withAux(
      this,
      { implicit scope => implicit forward =>
        val (v, m) = a.forward.varAndMean(dim, true, true)
        (v, List(m))
      }
    )

}
case class Dropout(a: Variable, prob: Double, train: Boolean) extends Op {

  val params = List(
    a.zipBackward { case grad @ Grad(p, out) =>
      implicit val fw = grad.fw
      if (prob > 0.0)  { 
        val mask = value.forwardAux._2.head
        out.addcmulSelf(p, mask, 1d)
      }
      else {
        out += p
      }
    }
  )

  val value =
    Variable.withAux(
      this,
      { implicit scope => implicit forward =>
        val av = a.forward
        if (prob <= 0d) {
          (av, Nil)
        } else {
          val mask = {
            val ones = STen.onesLike(av)
            ones.dropout_(prob, train)
            ones
          }
          (av * mask, List(mask))
        }
      }
    )

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
case object SumReduce extends Reduction {
  def asLong = 2L
}

case class MseLoss(
    input: Variable,
    target: STen,
    reduction: Reduction
) extends Op {

  val params = List(
    input.zipBackward { case grad @ Grad(p, out) =>
      implicit val fw = grad.fw
      Scope.root { implicit scope =>
        val tmp =
          STen.mse_loss_backward(
            p,
            input.forward,
            target.view(input.shape: _*),
            reduction.asLong
          )

        out += tmp
      }

    }
  )
  val value =
    Variable(
      this,
      { implicit scope => implicit forward =>
        val iv = input.forward
        assert(iv.numel == target.numel)
        STen.mse_loss(
          iv,
          target.view(input.shape: _*),
          reduction.asLong
        )
      }
    )

}
case class SmoothL1Loss(
    input: Variable,
    target: STen,
    reduction: Reduction,
    beta: Double
) extends Op {

  val params = List(
    input.zipBackward { case grad @ Grad(p, out) =>
      implicit val fw = grad.fw
      Scope.root { implicit scope =>
        val tmp =
          STen.owned(
            ATen.smooth_l1_loss_backward_0(
              p.value,
              input.forward.value,
              target.view(input.shape: _*).value,
              reduction.asLong,
              beta
            )
          )

        out += tmp

      }
    }
  )
  val value =
    Variable(
      this,
      { implicit scope => implicit forward =>
        val iv = input.forward
        assert(iv.numel == target.numel)
        STen.owned(
          ATen.smooth_l1_loss_0(
            iv.value,
            target.view(input.shape: _*).value,
            reduction.asLong,
            beta
          )
        )
      }
    )
}
case class NllLoss(
    input: Variable,
    target: STen,
    weights: STen,
    // numClasses: Int,
    reduction: Reduction,
    ignore: Long
) extends Op {

  val params = List(
    input.zipBackward { case grad @ Grad(p, out) =>
      implicit val fw = grad.fw
      Scope.root { implicit scope =>
        val total_weight = value.forwardAux._2.head.value
        val tmp =
          STen.owned(
            ATen.nll_loss_backward(
              p.value,
              input.forward.value,
              target.value,
              Option(weights.value),
              reduction.asLong,
              ignore,
              total_weight
            )
          )
        out += tmp

      }
    }
  )

  val value =
    Variable.withAux(
      this,
      { implicit scope => implicit forward =>
        val iv = input.forward
        assert(
          iv.shape.size == 2,
          "Nll Loss assumes 2D input (samples x classes). Higher dimensions not implemented."
        )
        assert(
          target.sizes.size == 1,
          "Target should be a 1D tensor with [0,C-1] integers, C number of classes."
        )

        val (value1, total_weight) =
          ATen.nll_loss_forward(
            iv.value,
            target.value,
            Option(weights.value),
            reduction.asLong,
            ignore
          )
        (STen.owned(value1), List(STen.owned(total_weight)))

      }
    )

}

/** input: (N,T) where T>=1 are multiple independent tasks target: same shape as
  * input, float with in [0,1] posWeight: is (T)
  */
case class BinaryCrossEntropyWithLogitsLoss(
    input: Variable,
    target: STen,
    posWeights: Option[STen],
    reduction: Reduction
) extends Op {

  val params = List(
    input.zipBackward { case grad@Grad(p, out) =>
      implicit val fw= grad.fw
      Scope.root { implicit scope =>
        // -[ pos * y * (1 -sigmoid(x)) - (1 - y) sigmoid(x)] * grad

        val iv = input.forward
        val t = if (posWeights.isDefined) {
          val t = posWeights.get * target
          val t2 = t + 1.0
          t2 -= target
          t2 *= iv.sigmoid
          t2 -= t
          t2
        } else {
          val t = iv.sigmoid
          t -= target
          t
        }

        t *= p

        if (reduction == Mean) {
          t.*=(1d / iv.numel.toDouble)
        }

        out += t

      }
    }
  )

  val value =
    Variable(
      this,
      { implicit scope => implicit forward =>
        val iv = input.forward
        assert(
          iv.sizes == target.sizes,
          s"BinaryCrossEntropyWithLogitsLoss input and target have the same shape. Input ${iv.sizes}, target ${target.sizes} ."
        )
        STen.owned(
          ATen.binary_cross_entropy_with_logits(
            iv.value,
            target.value,
            None,
            posWeights.map(_.value),
            reduction.asLong
          )
        )
      }
    )

}

// // /** 1D convolution
// //   *
// //   * @param input
// //   *   batch x in_channels x L
// //   * @param weight
// //   *   out_channels x in_channels x kernel_size
// //   * @param bias
// //   *   out_channels
// //   * @return
// //   *   Variable with Tensor of size batch x out_channels x L' (length depends on
// //   *   stride/padding/dilation)
// //   */
// // case class Conv1D(
// //     scope: Scope,
// //     input: Variable,
// //     weight: Variable,
// //     bias: Variable,
// //     stride: Long,
// //     padding: Long,
// //     dilation: Long,
// //     groups: Long
// // ) extends Op {

// //   assert(input.shape.size == 3, "Input dimensions must be 3")
// //   assert(weight.shape.size == 3, "Weight dimensions must be 3")
// //   val batchSize = input.shape(0)
// //   val inputChannels = input.shape(1)
// //   val imageSize = input.shape(2)
// //   val kernelSize = weight.shape(2)
// //   val outChannels = weight.shape(0)
// //   assert(
// //     weight.shape(1) == inputChannels,
// //     "Weight 2nd dimension must have size equal to input channels (2nd dim of input) "
// //   )
// //   assert(
// //     bias.shape(0) == outChannels,
// //     "Number of biases must be the number of output channels"
// //   )

// //   override val params: List[(Variable, (STen, STen) => Unit)] = List(
// //     input.zipBackward { case Grad(p, out) =>
// //       val pSize = p.sizes
// //       val zeros = ATen.zeros(Array(inputChannels), p.options.value)
// //       val outputSizeWithoutExtraPadding =
// //         (pSize(2) - 1) * stride - 2 * padding + dilation * (kernelSize - 1) + 1
// //       val extraPadding = out.sizes.apply(2) - outputSizeWithoutExtraPadding
// //       val tmp = ATen.conv_transpose1d(
// //         p.value,
// //         weight.value.value,
// //         Some(zeros),
// //         Array(stride),
// //         Array(padding),
// //         Array(extraPadding),
// //         groups,
// //         Array(dilation)
// //       )
// //       ATen.add_out(out.value, out.value, tmp, 1d)
// //       tmp.release
// //       zeros.release()
// //     },
// //     weight.zipBackward { case Grad(p, out) =>
// //       val p_repeated =
// //         ATen.repeat_interleave_2(p.value, inputChannels / groups, 1)
// //       val p_repeated_size = p_repeated.sizes
// //       val p_repeated_viewed =
// //         ATen._unsafe_view(
// //           p_repeated,
// //           Array(p_repeated_size(0) * p_repeated_size(1), 1, p_repeated_size(2))
// //         )
// //       val input_viewed = ATen._unsafe_view(
// //         input.value.value,
// //         Array(1, batchSize * inputChannels, imageSize)
// //       )
// //       val zero = ATen
// //         .zeros(Array(p_repeated_viewed.sizes.apply(0)), p.options.value)
// //       val conv_0 = ATen.conv1d_0(
// //         input_viewed,
// //         p_repeated_viewed,
// //         Some(zero),
// //         Array(dilation),
// //         Array(padding),
// //         Array(stride),
// //         inputChannels * batchSize
// //       )
// //       val conv_0_sizes = conv_0.sizes
// //       val conv_1 = ATen._unsafe_view(
// //         conv_0,
// //         Array(
// //           batchSize,
// //           conv_0_sizes.apply(1) / batchSize,
// //           conv_0_sizes.apply(2)
// //         )
// //       )

// //       val conv_1_sum = ATen.sum_1(conv_1, Array(0L), false)
// //       val conv_1_sum_viewed =
// //         ATen._unsafe_view(
// //           conv_1_sum,
// //           Array(inputChannels / groups, outChannels, conv_1.sizes.apply(2))
// //         )
// //       val conv_1_sum_viewed_transposed = ATen.transpose(conv_1_sum_viewed, 0, 1)

// //       val conv_1_sum_viewed_transposed_narrowed =
// //         ATen.narrow_0(conv_1_sum_viewed_transposed, 2, 0, kernelSize)
// //       ATen.add_out(
// //         out.value,
// //         out.value,
// //         conv_1_sum_viewed_transposed_narrowed,
// //         1d
// //       )

// //       conv_1_sum_viewed_transposed_narrowed.release()
// //       conv_1_sum_viewed_transposed.release
// //       conv_1_sum_viewed.release
// //       conv_1_sum.release
// //       conv_1.release
// //       conv_0.release
// //       input_viewed.release()
// //       p_repeated_viewed.release
// //       p_repeated.release

// //     },
// //     bias.zipBackward { case Grad(p, out) =>
// //       val p2 = ub(p.value, List(out.sizes.toList.head, 1)).getOrElse(p.value)
// //       val p3 = ATen._unsafe_view(p2, out.sizes.toArray)
// //       ATen.add_out(out.value, out.value, p3, 1d)
// //       if (p2 != p.value) {
// //         p2.release()
// //       }
// //       p3.release()
// //     }
// //   )

// //   val value =
// //     Variable(
// //       this, {
// //         STen.owned(
// //           ATen.conv1d_0(
// //             input.value.value,
// //             weight.value.value,
// //             Some(bias.value.value),
// //             Array(stride),
// //             Array(padding),
// //             Array(dilation),
// //             groups
// //           )
// //         )
// //       }
// //     )
// // }

/** 1D/2D/3D convolution
  *
  * @param input
  *   batch x in_channels x height x width
  * @param weight
  *   out_channels x in_channels x kernel_size x kernel_size
  * @param bias
  *   out_channels
  * @return
  *   Variable with Tensor of size batch x out_channels x L' (length depends on
  *   stride/padding/dilation)
  */
case class Convolution(
    scope: Scope,
    input: Variable,
    weight: Variable,
    bias: Variable,
    stride: Array[Long],
    padding: Array[Long],
    dilation: Array[Long],
    transposed: Boolean,
    outputPadding: Array[Long],
    groups: Long
) extends Op {

  // assert(input.shape.size == 4, "Input dimensions must be 4")
  // assert(weight.shape.size == 4, "Weight dimensions must be 4")
  // val batchSize = input.shape(0)
  // val inputChannels = input.shape(1)
  // val imageHeight = input.shape(2)
  // val imageWidth = input.shape(3)
  // val kernelSize = weight.shape(2)
  // val outChannels = weight.shape(0)

  override val joinedBackward: Option[Scope => JoinedGradHelper => Unit] =
    Some({ implicit scope => helper =>
      val p = helper.p
      implicit val fw= helper.forwardCache

      val (tmp0, tmp1, tmp2) = ATen.convolution_backward(
        p.value,
        input.forward.value,
        weight.forward.value,
        Some(bias.shape.toArray),
        stride,
        padding,
        dilation,
        transposed,
        outputPadding,
        groups,
        Array(true, true, true)
      )
      if (needsGrad(input)) {
        val outI = helper.partialDerivative(input)(scope)
        ATen.add_out(outI.value, outI.value, tmp0, 1d)
      }
      if (needsGrad(weight)) {
        val outW = helper.partialDerivative(weight)(scope)
        ATen.add_out(outW.value, outW.value, tmp1, 1d)
      }

      if (needsGrad(bias)) {
        val outB = helper.partialDerivative(bias)(scope)
        ATen.add_out(outB.value, outB.value, tmp2, 1d)
      }
      tmp0.release
      tmp1.release
      tmp2.release

    })

  override val params = List(
    input.missingBackward,
    weight.missingBackward,
    bias.missingBackward
  )

  // Tensor input,Tensor weight,scala.Option<Tensor> bias,long[] stride,long[] padding,long[] dilation,boolean transposed,long[] output_padding,long groups

  val value =
    Variable(
      this,
      { implicit scope => implicit forward =>
        STen.owned(
          ATen.convolution(
            input.forward.value,
            weight.forward.value,
            Some(bias.forward.value),
            stride,
            padding,
            dilation,
            transposed,
            outputPadding,
            groups
          )
        )
      }
    )
}

/** 1D max pooling
  *
  * @param input
  *   batch x in_channels x L
  */
case class MaxPool1D(
    scope: Scope,
    input: Variable,
    kernelSize: Long,
    stride: Long = 1,
    padding: Long = 0,
    dilation: Long = 1
) extends Op {

  override val params = List(
    input.zipBackward { case grad @ Grad(p, out) =>
      implicit val fw = grad.fw
      Scope.root { implicit scope =>
        assert(input.shape.size == 3, "Input dimensions must be 3")
        val (inputV, auxV) = value.forwardAux
        val mask = auxV.head.value
        import lamp.util.syntax
        val zeros = ATen.zeros_like(out.value, inputV.options.value)
        val p_flatten = ATen.flatten(p.value, 0, 1)
        val mask_flatten = ATen.flatten(mask, 0, 1)
        val zeros_flatten = ATen.flatten(zeros, 0, 1)
        val addeds = 0L until p_flatten.shape.apply(0) map { i =>
          val p_select = ATen.select(p_flatten, 0, i)
          val mask_select = ATen.select(mask_flatten, 0, i)
          val zeros_select = ATen.select(zeros_flatten, 0, i)
          val added = ATen.index_add_0(zeros_select, 0, mask_select, p_select)
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
    }
  )

  val value =
    Variable.withAux(
      this,
      { implicit scope => implicit forward =>
        val (output, mask) = ATen.max_pool1d_with_indices(
          input.forward.value,
          Array(kernelSize),
          Array(stride),
          Array(padding),
          Array(dilation),
          false
        )
        STen.owned(output) -> List(STen.owned(mask))

      }
    )
}

/** 2D max pooling
  *
  * @param input
  *   batch x in_channels x h x w
  */
case class MaxPool2D(
    scope: Scope,
    input: Variable,
    kernelSize: Long,
    stride: Long,
    padding: Long,
    dilation: Long
) extends Op {

  override val params = List(
    input.zipBackward { case grad @ Grad(p, out) =>
      implicit val fw = grad.fw
      Scope.root { implicit scope =>
        val mask = value.forwardAux._2.head
        val tmp = STen.owned(
          ATen.max_pool2d_with_indices_backward(
            p.value,
            input.forward.value,
            Array(kernelSize),
            Array(stride),
            Array(padding),
            Array(dilation),
            false,
            mask.value
          )
        )
        out += tmp
      }

    }
  )

  val value =
    Variable.withAux(
      this,
      { implicit scope => implicit forward =>
        assert(input.shape.size == 4, "Input dimensions must be 4")
        val (output, mask) = ATen.max_pool2d_with_indices(
          input.forward.value,
          Array(kernelSize),
          Array(stride),
          Array(padding),
          Array(dilation),
          false
        )
        STen.owned(output) -> List(STen.owned(mask))
      }
    )
}
