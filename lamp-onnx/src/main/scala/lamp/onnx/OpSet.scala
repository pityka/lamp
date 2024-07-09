package lamp.onnx

import lamp.autograd._
import onnx.{onnx => ox}
import java.util.UUID
import lamp.STen
import lamp.SinglePrecision
import lamp.DoublePrecision
import lamp.Scope
import lamp.HalfPrecision

trait NameMap {
  def apply(u: AnyRef): String
}

case class Converted(
    node: ox.NodeProto,
    constants: Seq[ox.TensorProto] = Nil
) {
  def withInput(s: Seq[String] => Seq[String]) =
    copy(node = node.withInput(s(node.input)))
  def addConstant(t: ox.TensorProto) = copy(constants = t +: constants)
  def appendInput(t: ox.TensorProto) =
    withInput(old => old :+ t.name.get).addConstant(t)
}

object Ops {
  private[lamp] val ComMicrosoft = "com.microsoft"
  def apply(
      output: VariableNonConstant,
      opType: String,
      attributes: Seq[ox.AttributeProto] = Nil,
      domain: Option[String] = None
  )(makeName: NameMap): Converted =
    Converted(
      ox.NodeProto(
        name = Some(makeName(output.id)),
        output = List(makeName(output.id)),
        opType = Some(opType),
        input = output.op.get.params.map(v => makeName(v._1.id)),
        attribute = attributes,
        domain = domain
      )
    )

  def tensorFromLongVec(vec: Seq[Long]): ox.TensorProto = {
    val name = UUID.randomUUID().toString.replace("-", "_")
    ox.TensorProto(
      name = Some(name),
      docString = Some(
        "Literal constant converted to a tensor to match ONNX operator signatures."
      ),
      dims = List(vec.length),
      dataType = Some(ox.TensorProto.DataType.INT64.index),
      int64Data = vec
    )
  }
  def tensorFromDoubleVec(vec: Seq[Double], `type`: Byte): ox.TensorProto = {
    val name = UUID.randomUUID().toString.replace("-", "_")
    `type` match {
      case 4 =>
        ox.TensorProto(
          name = Some(name),
          docString = Some(
            "Literal constant converted to a tensor to match ONNX operator signatures."
          ),
          dims = List(vec.length),
          dataType = Some(ox.TensorProto.DataType.INT64.index),
          int64Data = vec.map(_.toLong)
        )
      case 6 =>
        ox.TensorProto(
          name = Some(name),
          docString = Some(
            "Literal constant converted to a tensor to match ONNX operator signatures."
          ),
          dims = List(vec.length),
          dataType = Some(ox.TensorProto.DataType.FLOAT.index),
          floatData = vec.map(_.toFloat)
        )
      case 7 =>
        ox.TensorProto(
          name = Some(name),
          docString = Some(
            "Literal constant converted to a tensor to match ONNX operator signatures."
          ),
          dims = List(vec.length),
          dataType = Some(ox.TensorProto.DataType.DOUBLE.index),
          doubleData = vec
        )
    }
  }
  def tensorFromDoubleScalar(d: Double, `type`: Byte): ox.TensorProto = {
    val name = UUID.randomUUID().toString.replace("-", "_")
    `type` match {
      case 4 =>
        ox.TensorProto(
          name = Some(name),
          docString = Some(
            "Literal constant converted to a tensor to match ONNX operator signatures."
          ),
          dataType = Some(ox.TensorProto.DataType.INT64.index),
          int64Data = List(d.toLong)
        )
      case 6 =>
        ox.TensorProto(
          name = Some(name),
          docString = Some(
            "Literal constant converted to a tensor to match ONNX operator signatures."
          ),
          dataType = Some(ox.TensorProto.DataType.FLOAT.index),
          floatData = List(d.toFloat)
        )
      case 7 =>
        ox.TensorProto(
          name = Some(name),
          docString = Some(
            "Literal constant converted to a tensor to match ONNX operator signatures."
          ),
          dataType = Some(ox.TensorProto.DataType.DOUBLE.index),
          doubleData = List(d)
        )
    }

  }
  def tensorFromIntScalar(d: Int): ox.TensorProto = {
    val name = UUID.randomUUID().toString.replace("-", "_")
    ox.TensorProto(
      name = Some(name),
      docString = Some(
        "Literal constant converted to a tensor to match ONNX operator signatures."
      ),
      dataType = Some(ox.TensorProto.DataType.INT64.index),
      int64Data = List(d.toLong)
    )
  }
  def tensorFromLongScalar(d: Long): ox.TensorProto = {
    val name = UUID.randomUUID().toString.replace("-", "_")
    ox.TensorProto(
      name = Some(name),
      docString = Some(
        "Literal constant converted to a tensor to match ONNX operator signatures."
      ),
      dataType = Some(ox.TensorProto.DataType.INT64.index),
      int64Data = List(d)
    )
  }
  def tensorFromBooleanScalar(d: Boolean): ox.TensorProto = {
    val name = UUID.randomUUID().toString.replace("-", "_")
    ox.TensorProto(
      name = Some(name),
      docString = Some(
        "Literal constant converted to a tensor to match ONNX operator signatures."
      ),
      dataType = Some(ox.TensorProto.DataType.BOOL.index),
      int32Data = List(if (d) 1 else 0)
    )
  }
  def tensorFromSTen(d: STen): ox.TensorProto = {
    val name = UUID.randomUUID().toString.replace("-", "_")
    ox.TensorProto(
      name = Some(name),
      docString = Some(
        "Literal or constant converted to a tensor to match ONNX operator signatures."
      ),
      dims = d.shape,
      dataType = Scope.root { implicit scope =>
        d.options.scalarTypeByte match {
          case 4 => Some(ox.TensorProto.DataType.INT64.index)
          case 6 => Some(ox.TensorProto.DataType.FLOAT.index)
          case 7 => Some(ox.TensorProto.DataType.DOUBLE.index)
        }
      },
      rawData = Some(tensorAsByteString(d))
    )

  }

  def attr(name: String, value: Long) =
    ox.AttributeProto(
      name = Some(name),
      `type` = Some(ox.AttributeProto.AttributeType.INT),
      i = Some(value)
    )
  def attr(name: String, value: Float) =
    ox.AttributeProto(
      name = Some(name),
      `type` = Some(ox.AttributeProto.AttributeType.FLOAT),
      f = Some(value)
    )
  def attrLongSeq(name: String, value: Seq[Long]) =
    ox.AttributeProto(
      name = Some(name),
      `type` = Some(ox.AttributeProto.AttributeType.INTS),
      ints = value
    )
}

trait OpSet {
  def translate(m: NameMap, op: VariableNonConstant): Seq[Converted]
}

object DefaultOpSet extends DefaultOpSet1

trait DefaultOpSet1 extends OpSet {
  def translate(nm: NameMap, out: VariableNonConstant): Seq[Converted] =
    out.op.get match {
      case _: Transpose     => Ops(out, "Transpose")(nm) :: Nil
      case _: Add           => Ops(out, "Add")(nm) :: Nil
      case _: Minus         => Ops(out, "Sub")(nm) :: Nil
      case _: Mult          => Ops(out, "Mul")(nm) :: Nil
      case _: Div           => Ops(out, "Div")(nm) :: Nil
      case _: MatMul        => Ops(out, "MatMul")(nm) :: Nil
      case _: BatchedMatMul => Ops(out, "MatMul")(nm) :: Nil
      case _: Exp           => Ops(out, "Exp")(nm) :: Nil
      case _: Log           => Ops(out, "Log")(nm) :: Nil
      case _: Sin           => Ops(out, "Sin")(nm) :: Nil
      case _: Cos           => Ops(out, "Cos")(nm) :: Nil
      case _: Tan           => Ops(out, "Tan")(nm) :: Nil
      case _: Tanh          => Ops(out, "Tanh")(nm) :: Nil
      case _: ArcTan        => Ops(out, "Atan")(nm) :: Nil
      case _: Sigmoid       => Ops(out, "Sigmoid")(nm) :: Nil
      case _: Relu          => Ops(out, "Relu")(nm) :: Nil
      case _: Gelu =>
        Ops(out, "Gelu", domain = Some(Ops.ComMicrosoft))(nm) :: Nil
      case _: Pow => Ops(out, "Pow")(nm) :: Nil
      case o: View =>
        Ops(out, "Reshape")(nm)
          .appendInput(Ops.tensorFromLongVec(o.shape.toSeq)) :: Nil
      case o: Reshape =>
        Ops(out, "Reshape")(nm)
          .appendInput(Ops.tensorFromLongVec(o.shape.toSeq)) :: Nil
      case op: Concatenate =>
        Ops(out, "Concat", List(Ops.attr("axis", op.dim)))(nm) :: Nil

      case op: ArgMax =>
        Ops(
          out,
          "ArgMax",
          List(
            Ops.attr("axis", op.dim),
            Ops.attr("keepdims", if (op.keepDim) 1L else 0L)
          )
        )(nm) :: Nil
      case op: OneHot =>
        Ops(out, "OneHot")(nm)
          .appendInput(Ops.tensorFromIntScalar(op.numClasses))
          .appendInput(
            Ops.tensorFromDoubleVec(
              List(0d, 1d),
              Scope.root { implicit scope => op.value.options.scalarTypeByte }
            )
          ) :: Nil

      case op: ConstAdd =>
        Ops(out, "Add")(nm).appendInput(
          Ops.tensorFromDoubleScalar(
            op.b,
            Scope.root { implicit scope =>
              op.a.options.scalarTypeByte
            }
          )
        ) :: Nil

      case op: ConstMult =>
        Ops(out, "Mul")(nm).appendInput(
          Ops.tensorFromDoubleScalar(
            op.b,
            Scope.root { implicit scope =>
              op.a.options.scalarTypeByte
            }
          )
        ) :: Nil

      case op: Sum =>
        val axes =
          (if (op.dim.nonEmpty)
             List(
               Ops.attrLongSeq(
                 "axes",
                 op.dim.map(
                   _.toLong
                 )
               )
             )
           else Nil)
        val keepDim = List(Ops.attr("keepdims", if (op.keepDim) 1L else 0L))
        Ops(
          out,
          "ReduceSum",
          attributes = keepDim ++ axes
        )(nm) :: Nil
      case op: ExpandAs =>
        Ops(out, "Expand")(nm)
          .appendInput(Ops.tensorFromLongVec(op.as.shape)) :: Nil

      case op: PowConst =>
        Ops(out, "Pow")(nm).appendInput(
          Ops.tensorFromDoubleScalar(
            op.exponent,
            Scope.root { implicit scope =>
              op.a.options.scalarTypeByte
            }
          )
        ) :: Nil
      case op: LogSoftMax =>
        Ops(out, "LogSoftmax", attributes = List(Ops.attr("axis", op.dim)))(
          nm
        ) :: Nil
      case op: Mean =>
        val axes =
          (if (op.dim.nonEmpty)
             List(
               Ops.attrLongSeq(
                 "axes",
                 op.dim.map(
                   _.toLong
                 )
               )
             )
           else Nil)
        val keepDim = List(Ops.attr("keepdims", if (op.keepDim) 1L else 0L))
        Ops(
          out,
          "ReduceMean",
          attributes = keepDim ++ axes
        )(nm) :: Nil
      case op: Dropout =>
        Ops(out, "Dropout")(nm)
          .appendInput(Ops.tensorFromDoubleScalar(op.prob, 7))
          .appendInput(Ops.tensorFromBooleanScalar(op.train)) :: Nil
      case op: Flatten =>
        assert(op.endDim == op.input.shape.length - 1 || op.endDim == -1)
        Ops(out, "Flatten", attributes = List(Ops.attr("axis", op.startDim)))(
          nm
        ) :: Nil

      case op: Convolution if !op.transposed =>
        Ops(
          out,
          "Conv",
          attributes = List(
            Ops.attrLongSeq("dilations", op.dilation.toList),
            Ops.attrLongSeq("pads", op.padding.toList.flatMap(x => List(x,x))),
            Ops.attrLongSeq("strides", op.stride.toList),
            Ops.attr("group", op.groups)
          )
        )(nm) :: Nil
      case op: Convolution if op.transposed =>
        Ops(
          out,
          "ConvTranspose",
          attributes = List(
            Ops.attrLongSeq("dilations", op.dilation.toList),
            Ops.attrLongSeq("pads", op.padding.toList.flatMap(x => List(x,x))),
            Ops.attrLongSeq("strides", op.stride.toList),
            Ops.attrLongSeq("output_padding", op.outputPadding.toList),
            Ops.attr("group", op.groups)
          )
        )(nm) :: Nil

      case op: MaxPool1D =>
        Ops(
          out,
          "MaxPool",
          attributes = List(
            Ops.attrLongSeq("dilations", List(op.dilation)),
            Ops.attrLongSeq("kernel_shape", List(op.kernelSize)),
            Ops.attrLongSeq("pads", List(op.padding, op.padding)),
            Ops.attrLongSeq("strides", List(op.stride))
          )
        )(nm) :: Nil
      case op: MaxPool2D =>
        Ops(
          out,
          "MaxPool",
          attributes = List(
            Ops.attrLongSeq("dilations", List(op.dilation, op.dilation)),
            Ops.attrLongSeq("kernel_shape", List(op.kernelSize, op.kernelSize)),
            Ops.attrLongSeq(
              "pads",
              List(op.padding, op.padding, op.padding, op.padding)
            ),
            Ops.attrLongSeq("strides", List(op.stride, op.stride))
          )
        )(nm) :: Nil
      case op: AvgPool2D =>
        Ops(
          out,
          "AveragePool",
          attributes = List(
            Ops.attrLongSeq("kernel_shape", List(op.kernelSize, op.kernelSize)),
            Ops.attr("count_include_pad", 1),
            Ops.attrLongSeq(
              "pads",
              List(op.padding, op.padding, op.padding, op.padding)
            ),
            Ops.attrLongSeq("strides", List(op.stride, op.stride))
          )
        )(nm) :: Nil
      case op: BatchNorm =>
        Ops(
          out,
          "BatchNormalization",
          attributes = List(
            Ops.attr("momentum", op.momentum.toFloat)
          )
        )(nm)
          .appendInput(Ops.tensorFromSTen(op.runningMean))
          .appendInput(Ops.tensorFromSTen(op.runningVar)) :: Nil
      case op: BatchNorm2D =>
        Ops(
          out,
          "BatchNormalization",
          attributes = List(
            Ops.attr("momentum", op.momentum.toFloat)
          )
        )(nm)
          .appendInput(Ops.tensorFromSTen(op.runningMean))
          .appendInput(Ops.tensorFromSTen(op.runningVar)) :: Nil

      case op: CastToPrecision =>
        Ops(
          out,
          "Cast",
          attributes = List(
            Ops.attr(
              "to",
              op.precision match {
                case HalfPrecision =>
                  ox.TensorProto.DataType.FLOAT16.index.toLong
                case SinglePrecision =>
                  ox.TensorProto.DataType.FLOAT.index.toLong
                case DoublePrecision =>
                  ox.TensorProto.DataType.DOUBLE.index.toLong
              }
            )
          )
        )(nm) :: Nil

      case op: Select =>
        Ops(
          out,
          "Gather",
          attributes = List(
            Ops.attr("axis", op.dim)
          )
        )(nm).appendInput(Ops.tensorFromLongScalar(op.index)) :: Nil
      case op: IndexSelect =>
        Ops(
          out,
          "Gather",
          attributes = List(
            Ops.attr("axis", op.dim)
          )
        )(nm).appendInput(Ops.tensorFromSTen(op.index.value)) :: Nil
      case op: Assign =>
        Converted(
          ox.NodeProto(
            name = Some(nm(op.value.id)),
            output = List(nm(op.value.id)),
            opType = Some("Identity"),
            input = op.params.takeRight(1).map(v => nm(v._1.id))
          )
        ) :: Nil
      case op: Stack =>
        val unsqueezes = op.a.map { input =>
          val name = UUID.randomUUID()
          ox.NodeProto(
            name = Some(nm(name)),
            output = List(nm(name)),
            opType = Some("Unsqueeze"),
            input = List(nm(input.id)),
            attribute = List(Ops.attrLongSeq("axes", List(op.dim)))
          )
        }
        val cat = List(
          ox.NodeProto(
            name = Some(nm(op.value.id)),
            output = List(nm(op.value.id)),
            opType = Some("Concat"),
            input = unsqueezes.map(_.name.get),
            attribute = List(Ops.attr("axis", op.dim))
          )
        )
        (unsqueezes ++ cat).map(Converted(_))

      // case _: IndexFill                        => Ops(out, "Transpose")(nm)
      // case _: IndexAdd                         => Ops(out, "Transpose")(nm)
      // case _: RepeatInterleave                 => Ops(out, "Transpose")(nm)
      // case _: MaskFill                         => Ops(out, "Transpose")(nm)
      // case _: ScatterAdd                       => Ops(out, "Transpose")(nm)

      // Skipped Ops:
      // case _: EqWhere                          => Ops(out, "Transpose")(nm)
      // case _: Embedding                        => Ops(out, "Transpose")(nm):: Nil
      // case _: Variance                         => Ops(out, "Transpose")(nm) :: Nil
      // case _: MseLoss                          => Ops(out, "Transpose")(nm) :: Nil
      // case _: L1Loss                           => Ops(out, "Transpose")(nm) :: Nil
      // case _: NllLoss                          => Ops(out, "Transpose")(nm) :: Nil
      case _ =>
        throw new RuntimeException(
          out.op.get.toString + " has no conversion to ONNX"
        )
    }
}
