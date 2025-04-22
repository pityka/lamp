package lamp.autograd

import lamp.STen
import lamp.Scope
import aten.ATen
/** 2D avg pooling
  *
  * @param input
  *   batch x in_channels x h x w
  */
case class AvgPool2D(
    scope: Scope,
    input: Variable,
    kernelSize: Long,
    stride: Long,
    padding: Long
) extends Op {

  override val params = List(
    input.zipBackward { case grad@Grad(p, out) =>
implicit val fw = grad.fw
      Scope.root { implicit scope =>
        val tmp = STen.owned(
          ATen.avg_pool2d_backward(
            p.value,
            input.forward.value,
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
      { implicit scope => implicit forward =>
        assert(input.shape.size == 4, "Input dimensions must be 4")
        STen.owned(
          ATen.avg_pool2d(
            input.forward.value,
            Array(kernelSize),
            Array(stride),
            Array(padding),
            false,
            true,
            Long.MinValue
          )
        )
      }
    )
}

case class Flatten(input: Variable, startDim: Int, endDim: Int) extends Op {

  override val params = List(
    input.zipBackward { case Grad(p, out) =>
      Scope.root { implicit scope => out += p.view(out.shape: _*) }
    }
  )

  val value =
    Variable(
      this,
      { implicit scope => implicit forward =>
        assert(input.shape.size >= 2, "Input dimensions must be >= 2")
        input.forward.flatten(startDim, endDim)
      }
    )
}
case class FlattenLastDimensions(input: Variable, dims: Int) extends Op {

  override val params = List(
    input.zipBackward { case Grad(p, out) =>
      Scope.root { implicit scope => out += p.view(out.shape: _*) }
    }
  )

  val value =
    Variable(
      this,
      { implicit scope => implicit forward =>
        assert(input.shape.size >= 2, "Input dimensions must be >= 2")
        val iv = input.forward 
        val startDim = iv.shape.size - dims 
        val endDim = -1
        
        input.forward.flatten(startDim, endDim)
      }
    )
}

// /* 0-th dimension has samples. Everything else is flattened out into features. */
// case class BatchNorm(
//     scope: Scope,
//     input: Variable,
//     weight: Variable,
//     bias: Variable,
//     runningMean: STen,
//     runningVar: STen,
//     training: Boolean,
//     momentum: Double,
//     eps: Double
// ) extends Op {

//   val input_flattened = ATen.flatten(input.value.value, 1, input.shape.size - 1)
//   val expectedShape = List(input_flattened.shape.last)
//   assert(
//     expectedShape == weight.shape,
//     s"Expected $expectedShape got weight shape ${weight.shape}"
//   )
//   assert(
//     expectedShape == bias.shape,
//     s"Expected $expectedShape got bias shape ${bias.shape}"
//   )
//   assert(
//     expectedShape == runningMean.shape,
//     s"Expected $expectedShape got runningMean shape ${runningMean.shape}"
//   )
//   assert(
//     expectedShape == runningVar.shape,
//     s"Expected $expectedShape got runningVar shape ${runningVar.shape}"
//   )

//   val (output, saveMean, saveInvstd) = ATen.native_batch_norm(
//     input_flattened,
//     Option(weight.value.value),
//     Option(bias.value.value),
//     Option(runningMean.value),
//     Option(runningVar.value),
//     training,
//     momentum,
//     eps
//   )
//   val output_reshaped = ATen._unsafe_view(output, input.shape.toArray)

//   scope.register(saveMean)
//   scope.register(saveInvstd)
//   scope.register(input_flattened)
//   scope.register(output)

//   override val value: Variable =
//     Variable(this, STen.owned(output_reshaped))

//   override val params: List[(Variable, (STen, STen) => Unit)] = List(
//     input.zipBackward { case grad@Grad(p, out) =>
// implicit val fw = grad.fw
//       val flattened_p =
//         ATen.flatten(p.value, 1, p.shape.size - 1)
//       val (gradInput, a, b) = ATen.native_batch_norm_backward(
//         flattened_p,
//         input_flattened,
//         Option(weight.value.value),
//         Option(runningMean.value),
//         Option(runningVar.value),
//         Option(saveMean),
//         Option(saveInvstd),
//         training,
//         eps,
//         Array(true, false, false)
//       )
//       val gradInput_reshaped = ATen._unsafe_view(gradInput, out.shape.toArray)
//       ATen.add_out(out.value, out.value, gradInput_reshaped, 1d)
//       gradInput.release
//       a.release
//       b.release
//       flattened_p.release()
//       gradInput_reshaped.release
//     },
//     weight.zipBackward { case grad@Grad(p, out) =>
// implicit val fw = grad.fw
//       val flattened_p =
//         ATen.flatten(p.value, 1, p.shape.size - 1)
//       val (a, gradWeight, b) = ATen.native_batch_norm_backward(
//         flattened_p,
//         input_flattened,
//         Option(weight.value.value),
//         Option(runningMean.value),
//         Option(runningVar.value),
//         Option(saveMean),
//         Option(saveInvstd),
//         training,
//         eps,
//         Array(false, true, false)
//       )
//       val grad_reshaped = ATen._unsafe_view(gradWeight, out.shape.toArray)
//       ATen.add_out(out.value, out.value, grad_reshaped, 1d)
//       gradWeight.release
//       grad_reshaped.release
//       a.release
//       b.release
//       flattened_p.release()
//     },
//     bias.zipBackward { case grad@Grad(p, out) =>
// implicit val fw = grad.fw
//       val flattened_p =
//         ATen.flatten(p.value, 1, p.shape.size - 1)
//       val tmp = ub(flattened_p, out.shape).getOrElse(flattened_p)
//       ATen.add_out(out.value, out.value, tmp, 1d)
//       if (tmp != flattened_p) {
//         tmp.release
//       }
//       flattened_p.release
//     }
//   )
// }
case class LayerNormOp(
    scope: Scope,
    input: Variable,
    weight: Option[Variable],
    bias: Option[Variable],
    normalizedShape: List[Long],
    eps: Double
) extends Op {

// static inline std::tuple<Tensor,Tensor,Tensor> native_layer_norm_backward(const Tensor & grad_out, const Tensor & input, IntArrayRef normalized_shape, const Tensor & mean, const Tensor & rstd, const c10::optional<Tensor> & weight, const c10::optional<Tensor> & bias, std::array<bool,3> output_mask);

  def make(implicit scope: Scope) = {}

  override val value: Variable =
    Variable.withAux(
      this,
      { implicit scope => implicit forward =>
        val (output, mean, rstd) = ATen.native_layer_norm(
          input.forward.value,
          normalizedShape.toArray,
          weight.map(_.forward.value),
          bias.map(_.forward.value),
          eps
        )
        STen.owned(output) -> List(STen.owned(mean), STen.owned(rstd))
      }
    )

  override val joinedBackward = Some { implicit scope => helper =>
    val p = helper.p
    implicit val fw = helper.forwardCache
    val aux = value.forwardAux._2

    val mean = aux(0).value
    val rstd = aux(1).value


    val (gradInput, a, b) = ATen.native_layer_norm_backward(
      p.value,
      (input.forward).value,
      normalizedShape.toArray,
      (mean),
      (rstd),
      (weight.map(_.forward)).map(_.value),
      (bias.map(_.forward)).map(_.value),
      Array(true, weight.isDefined, bias.isDefined)
    )

    if (needsGrad(input)) {
      val inputOut = helper.partialDerivative(input)(scope)
      ATen.add_out(inputOut.value, inputOut.value, gradInput, 1d)
    }

    if (weight.exists(needsGrad)) {
      val w2 = weight.map(w => helper.partialDerivative(w)(scope))
      w2.foreach(out => ATen.add_out(out.value, out.value, a, 1d))
    }

    if (bias.exists(needsGrad)) {
      val b2 = bias.map(b => helper.partialDerivative(b)(scope))
      b2.foreach(out => ATen.add_out(out.value, out.value, b, 1d))
    }

    gradInput.release
    a.release
    b.release

  }

  override val params = List(input.missingBackward)

}

// /** Batch Norm 2D 0-th dimension are samples. 1-th are features, everything else
//   * is averaged out.
//   */
// case class BatchNorm2D(
//     scope: Scope,
//     input: Variable,
//     weight: Variable,
//     bias: Variable,
//     runningMean: STen,
//     runningVar: STen,
//     training: Boolean,
//     momentum: Double,
//     eps: Double
// ) extends Op {

//   val inputShape = input.shape
//   assert(inputShape.size >= 3, "Expected 3D or 4D tensor")
//   val expectedShape = List(inputShape(1))
//   assert(
//     expectedShape == weight.shape,
//     s"Expected $expectedShape got weight shape ${weight.shape}"
//   )
//   assert(
//     expectedShape == bias.shape,
//     s"Expected $expectedShape got bias shape ${bias.shape}"
//   )
//   assert(
//     expectedShape == runningMean.shape,
//     s"Expected $expectedShape got runningMean shape ${runningMean.shape}"
//   )
//   assert(
//     expectedShape == runningVar.shape,
//     s"Expected $expectedShape got runningVar shape ${runningVar.shape}"
//   )

//   val (output, saveMean, saveInvstd) = ATen.native_batch_norm(
//     input.value.value,
//     Option(weight.value.value),
//     Option(bias.value.value),
//     Option(runningMean.value),
//     Option(runningVar.value),
//     training,
//     momentum,
//     eps
//   )
//   scope.register(saveMean)
//   scope.register(saveInvstd)
//   override val value: Variable =
//     Variable(this, STen.owned(output))

//   override val params: List[(Variable, (STen, STen) => Unit)] = List(
//     input.zipBackward { case grad@Grad(p, out) =>
// implicit val fw = grad.fw
//       val (gradInput, a, b) = ATen.native_batch_norm_backward(
//         p.value,
//         input.value.value,
//         Option(weight.value.value),
//         Option(runningMean.value),
//         Option(runningVar.value),
//         Option(saveMean),
//         Option(saveInvstd),
//         training,
//         eps,
//         Array(true, false, false)
//       )
//       val gradInput_reshaped =
//         ATen._unsafe_view(gradInput, out.shape.toArray)
//       ATen.add_out(out.value, out.value, gradInput_reshaped, 1d)
//       gradInput.release
//       a.release
//       b.release
//       gradInput_reshaped.release
//     },
//     weight.zipBackward { case grad@Grad(p, out) =>
// implicit val fw = grad.fw
//       val (a, gradWeight, b) = ATen.native_batch_norm_backward(
//         p.value,
//         input.value.value,
//         Option(weight.value.value),
//         Option(runningMean.value),
//         Option(runningVar.value),
//         Option(saveMean),
//         Option(saveInvstd),
//         training,
//         eps,
//         Array(false, true, false)
//       )
//       val grad_reshaped = ATen._unsafe_view(gradWeight, out.shape.toArray)
//       ATen.add_out(out.value, out.value, grad_reshaped, 1d)
//       gradWeight.release
//       grad_reshaped.release
//       a.release
//       b.release
//     },
//     bias.zipBackward { case grad@Grad(p, out) =>
// implicit val fw = grad.fw
//       val tmp = ub(p.value, (out.shape ++ (1 to p.shape.size - 2).map(_ => 1L)))
//         .getOrElse(p.value)
//       val tmp_viewed = ATen._unsafe_view(
//         tmp,
//         out.shape.toArray
//       )
//       ATen.add_out(out.value, out.value, tmp_viewed, 1d)
//       if (p.value != tmp) {
//         tmp.release
//       }
//       tmp_viewed.release
//     }
//   )
// }
case class Embedding(input: STen, weight: Variable) extends Op {

  override val params = List(
    weight.zipBackward { case grad@Grad(p, out) =>
implicit val fw = grad.fw
      Scope.root { implicit scope =>
        val tmp = STen.owned(
          ATen
            .embedding_backward(
              p.value,
              input.value,
              weight.shape.apply(0),
              0L,
              false,
              false
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
        assert(input.shape.size >= 1, "Input dimensions must be at least 1")
        assert(weight.shape.size == 2, "Weight must have 2 dimensions")
        try {
          STen.owned(
            ATen
              .embedding(
                weight.forward.value,
                input.value,
                0L,
                false,
                false
              )
          )
        } catch {
          case e: Throwable =>
            println(weight.shape)
            println(input.toLongArray.min)
            println(input.toLongArray.max)
            throw e
        }
      }
    )

}

case class Cholesky(
    input: Variable
) extends Op {

  val params = List(
    input.zipBackward { case grad @ Grad(p, out) =>
      implicit val fw = grad.fw
      Scope.root { implicit scope =>
        // Ref: Pytorch FunctionsManual.cpp
        // https://arxiv.org/pdf/1602.07527.pdf
        val batch = input.shape.size == 3
        val size = input.shape.apply(input.shape.size - 2)
        val l = value.forward

        val g = p

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
    implicit scope => implicit forward => input.forward.choleskyLower
  )
}
case class CholeskySolve(
    b: Variable,
    factor: Variable,
    upper: Boolean
) extends Op {

  val params = List(
    b.zipBackward { case grad@Grad(p, out) =>
implicit val fw = grad.fw
      Scope.root { implicit scope =>
        val gB = p.choleskySolve(factor.forward, upper)

        out += gB

      }
    },
    factor.zipBackward { case grad @ Grad(p, out) =>
      implicit val fw = grad.fw
      Scope.root { implicit scope =>
        val factorV = factor.forward
        val gB = p.choleskySolve(factorV, upper)
        val batched = b.shape.size == 3
        val v = value.forward
        val tmp =
          if (batched) gB.bmm(v.transpose(-2, -1))
          else gB.mm(v.transpose(-2, -1))

        val tmp2 = tmp + tmp.transpose(-2, -1)

        val tmp4 = if (upper) {
          if (batched) factorV.bmm(tmp2)
          else factorV.mm(tmp2)
        } else {
          if (batched) tmp2.bmm(factorV)
          else tmp2.mm(factorV)
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
    implicit scope =>
      implicit forward =>
        b.forward
          .choleskySolve(factor.forward, upper)(
            scope
          )
  )
}

case class ElementWiseMinimum(a: Variable, b: Variable) extends Op {

  val params = List(
    a.zipBackward { case grad @ Grad(p, out) =>
      implicit val fw = grad.fw
      Scope.root { implicit scope =>
        val tmp = STen.zerosLike(out)
        val v = value.forwardAux._1
        val mask = a.forward.equ(v)
        out += tmp.maskedScatter(mask, p)
      }
    },
    b.zipBackward { case grad @ Grad(p, out) =>
      implicit val fw = grad.fw
      Scope.root { implicit scope =>
        val tmp = STen.zerosLike(out)
        val v = value.forwardAux._1
        val mask = a.forward.equ(v)
        val maskneg = mask.not
        out += tmp.maskedScatter(maskneg, p)
      }
    }
  )

  val value =
    Variable(
      this,
      implicit scope => implicit forward => a.forward.min(b.forward)
    )

}
case class ElementWiseMaximum(a: Variable, b: Variable) extends Op {

  val params = List(
    a.zipBackward { case grad @ Grad(p, out) =>
      implicit val fw = grad.fw
      Scope.root { implicit scope =>
        val tmp = STen.zerosLike(out)
        val v = value.forwardAux._1
        val mask = a.forward.equ(v)
        out += tmp.maskedScatter(mask, p)
      }
    },
    b.zipBackward { case grad @ Grad(p, out) =>
      implicit val fw = grad.fw
      Scope.root { implicit scope =>
        val tmp = STen.zerosLike(out)
        val v = value.forwardAux._1
        val mask = a.forward.equ(v)
        val maskneg = mask.not
        out += tmp.maskedScatter(maskneg, p)
      }
    }
  )

  val value =
    Variable(
      this,
      implicit scope => implicit forward => a.forward.max(b.forward)
    )

}

// case class ScaledDotProductAttention(
//     scope: Scope,
//     query: Variable,
//     key: Variable,
//     valueIn: Variable,
//     attentionBias: Option[STen],
//     isCausal: Boolean
// ) extends Op {

//   val value = Variable.withAux(
//     this,
//     { implicit scope => implicit forward =>
//       val (out, lse, cumsq, cumsk, maxq, maxk, philoxseed, philoxoffset) =
//         STen.scaledDotProductAttention(
//           query.forward,
//           key.forward,
//           valueIn.forward,
//           attentionBias,
//           isCausal
//         )
//       (
//         out,
//         List(
//           lse,
//           cumsq,
//           cumsk,
//           STen.fromLongArray(Array(maxq, maxk)),
//           philoxseed,
//           philoxoffset
//         )
//       )
//     }
//   )

//   val params =
//     List(
//       query.missingBackward,
//       key.missingBackward,
//       valueIn.missingBackward
//     )
//   override val joinedBackward = Some { _ => helper =>
//     val p = helper.p
//     implicit val fw = helper.forwardCache
//     val (out, aux) = value.forwardAux
//     val lse = aux(0)
//     val cumsq = aux(1)
//     val cumsk = aux(2)
//     val arr = aux(3).toLongArray
//     val maxq = arr(0)
//     val maxk = arr(1)
//     val philoxseed = aux(4)
//     val philoxoffset = aux(5)
//     Scope.root { implicit scope =>
//       val (gQ, gK, gV) = STen.scaledDotProductAttentionBackward(
//         p,
//         (query.forward),
//         (key.forward),
//         (valueIn.forward),
//         out,
//         attentionBias,
//         lse,
//         isCausal,
//         philoxseed,
//         philoxoffset,
//         cumsq,
//         cumsk,
//         maxq,
//         maxk
//       )
//       if (needsGrad(query)) {
//         val (outQ) = helper.partialDerivative(query)(scope)
//         outQ += (gQ)
//       }
//       if (needsGrad(key)) {
//         val (outK) = helper.partialDerivative(key)(scope)
//         outK += (gK)
//       }

//       if (needsGrad(valueIn)) {
//         val (outVIn) = helper.partialDerivative(valueIn)(scope)
//         outVIn += (gV)
//       }
//     }

//   }

// }

// case class Debug(
//     scope: Scope,
//     a: Variable,
//     callback: (STen, Boolean, Boolean) => Unit
// ) extends Op {

//   val params = List(
//     a.zipBackward { case grad@Grad(p, out) =>
// implicit val fw = grad.fw
//       Scope.root { implicit scope =>
//         val hasna = p.isnan.any.castToLong.toLongArray.apply(0) == 1
//         val hasbig = p.ge(1e6).any.castToLong.toLongArray.apply(0) == 1
//         callback(p, hasna, hasbig)

//       }
//       out += p
//     }
//   )

//   val value = Variable(this, a.value)

// }
