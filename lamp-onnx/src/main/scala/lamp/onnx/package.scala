package lamp
import _root_.onnx.{onnx => ox}
import lamp.autograd._
import _root_.onnx.onnx.TensorProto.DataType
import _root_.onnx.onnx.TensorShapeProto
import com.google.protobuf.ByteString
import java.nio.ByteBuffer
import java.io.File
import java.io.BufferedOutputStream
import java.io.FileOutputStream
import java.util.UUID

package object onnx {
  def serializeToFile(
      file: File,
      output: Variable,
      domain: String = "org.domain",
      modelDocString: String = "",
      opset: OpSet = DefaultOpSet
  )(infoFun: PartialFunction[Variable, VariableInfo]) = {
    val model = serialize(output, domain, modelDocString, opset)(infoFun)
    val bos = new BufferedOutputStream(new FileOutputStream(file))
    try {
      model.writeTo(bos)
    } finally {
      bos.close
    }
  }

  def tensorAsByteString(t: STen): ByteString = Scope.leak { implicit scope =>
    t.options.scalarTypeByte match {
      case 4 =>
        val array = t.toLongVec.toArray
        val bb = ByteBuffer
          .allocate(array.length * 8)
          .order(java.nio.ByteOrder.LITTLE_ENDIAN)
        var i = 0
        val n = array.length
        while (i < n) {
          bb.putLong(array(i))
          i += 1
        }
        bb.rewind()
        ByteString.copyFrom(bb)
      case 7 =>
        val array = t.toVec.toArray
        val bb = ByteBuffer
          .allocate(array.length * 8)
          .order(java.nio.ByteOrder.LITTLE_ENDIAN)
        var i = 0
        val n = array.length
        while (i < n) {
          bb.putDouble(array(i))
          i += 1
        }
        bb.rewind()
        ByteString.copyFrom(bb)
      case 6 =>
        val array = t.toFloatVec.toArray
        val bb = ByteBuffer
          .allocate(array.length * 4)
          .order(java.nio.ByteOrder.LITTLE_ENDIAN)
        var i = 0
        val n = array.length
        while (i < n) {
          bb.putFloat(array(i))
          i += 1
        }
        bb.rewind()
        ByteString.copyFrom(bb)
    }

  }

  def serialize(
      output: Variable,
      domain: String = "org.domain",
      modelDocString: String = "",
      opset: OpSet = DefaultOpSet
  )(infoFun: PartialFunction[Variable, VariableInfo]): ox.ModelProto = {

    def makeType(v: Variable) =
      ox.TypeProto(value =
        ox.TypeProto.Value.TensorType(value =
          ox.TypeProto.Tensor(
            elemType = Scope.leak { implicit scope =>
              v.value.options.scalarTypeByte match {
                case 4 => Some(DataType.INT64.index)
                case 6 => Some(DataType.FLOAT.index)
                case 7 => Some(DataType.DOUBLE.index)
              }
            },
            shape = Some(
              TensorShapeProto(dim =
                v.value.shape.map(shapeL =>
                  TensorShapeProto.Dimension(value =
                    TensorShapeProto.Dimension.Value.DimValue(shapeL)
                  )
                )
              )
            )
          )
        )
      )

    val graph = output.wengert.reverse
    val info = graph.collect(infoFun)

    val inputs = info.filter(_.input).map(_.variable.id)
    val nameMap = info.map { input => input.variable.id -> input.name }.toMap

    def makeName(u: UUID) =
      nameMap.get(u).getOrElse(u.toString.replaceAllLiterally("-", "_"))

    val namer = new NameMap {
      def apply(u: UUID): String = makeName(u)
    }

    val constantNodes = graph.collect {
      case x: ConstantWithoutGrad => x
    }
    val (inputNodes, nonInputConstantNodes) =
      constantNodes.partition(v => inputs.contains(v.id))

    val parameters = graph.collect {
      case x: ConstantWithGrad => x
    }
    val convertedNodes = graph.collect {
      case variable: VariableNonConstant =>
        opset.translate(namer, variable)

    }
    val nodes = convertedNodes.flatMap(_.map(_.node))
    val constants = convertedNodes.flatMap {
      _.flatMap {
        case Converted(_, constants) =>
          constants
      }
    }
    ox.ModelProto(
      irVersion = Some(ox.Version.IR_VERSION.index),
      producerName = Some("lamp"),
      domain = Some(domain),
      docString = Some(modelDocString),
      opsetImport = List(
        ox.OperatorSetIdProto(version = Some(12L)),
        ox.OperatorSetIdProto(
          domain = Some("com.microsoft"),
          version = Some(1L)
        )
      ),
      graph = Some(
        ox.GraphProto(
          name = Some("graph1"),
          node = nodes,
          initializer = constants ++ (nonInputConstantNodes ++ parameters)
            .filterNot(v => Scope.leak { implicit scope => v.options.isSparse })
            .map { variable =>
              ox.TensorProto(
                name = Some(makeName(variable.id)),
                docString = info
                  .find(_.variable.id == variable.id)
                  .map(_.docString),
                dims = variable.shape,
                dataType = Scope.leak { implicit scope =>
                  variable.options.scalarTypeByte match {
                    case 4 => Some(ox.TensorProto.DataType.INT64.index)
                    case 6 => Some(ox.TensorProto.DataType.FLOAT.index)
                    case 7 => Some(ox.TensorProto.DataType.DOUBLE.index)
                  }
                },
                rawData = Some(tensorAsByteString(variable.value))
              )
            },
          sparseInitializer = (nonInputConstantNodes ++ parameters)
            .filter(v => Scope.leak { implicit scope => v.options.isSparse })
            .map { variable =>
              Scope.leak {
                implicit scope =>
                  val coalesced = variable.value.coalesce
                  val values = coalesced.values
                  val indices = coalesced.indices
                  ox.SparseTensorProto(
                    values = Some(
                      ox.TensorProto(
                        name = Some(makeName(variable.id)),
                        docString = info
                          .find(_.variable.id == variable.id)
                          .map(_.docString),
                        dims = values.shape,
                        dataType = values.options.scalarTypeByte match {
                          case 4 => Some(ox.TensorProto.DataType.INT64.index)
                          case 6 => Some(ox.TensorProto.DataType.FLOAT.index)
                          case 7 => Some(ox.TensorProto.DataType.DOUBLE.index)
                        },
                        rawData = Some(tensorAsByteString(values))
                      )
                    ),
                    indices = Some(
                      ox.TensorProto(
                        dims = indices.shape,
                        dataType = indices.options.scalarTypeByte match {
                          case 4 => Some(ox.TensorProto.DataType.INT64.index)
                          case 6 => Some(ox.TensorProto.DataType.FLOAT.index)
                          case 7 => Some(ox.TensorProto.DataType.DOUBLE.index)
                        },
                        rawData = Some(tensorAsByteString(indices))
                      )
                    ),
                    dims = variable.shape
                  )
              }
            },
          input = inputNodes.map { variable =>
            ox.ValueInfoProto(
              name = Some(makeName(variable.id)),
              `type` = Some(makeType(variable)),
              docString =
                info.find(_.variable.id == variable.id).map(_.docString)
            )
          },
          output = List(
            ox.ValueInfoProto(
              name = Some(makeName(output.id)),
              `type` = Some(makeType(output)),
              docString = info.find(_.variable.id == output.id).map(_.docString)
            )
          )
        )
      )
    )
  }
  val a = ox.ModelProto()
}
