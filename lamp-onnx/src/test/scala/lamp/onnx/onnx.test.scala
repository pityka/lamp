package lamp.onnx

import org.scalatest.funsuite.AnyFunSuite

import _root_.lamp._
import _root_.lamp.autograd._
import java.io.File
import ai.onnxruntime._
// import org.saddle._
import scala.jdk.CollectionConverters._
import java.nio._

class OnnxSuite extends AnyFunSuite {

  def runModel(file: File, input: Map[String, STen], `type`: Byte)(implicit
      scope: Scope
  ) = {
    val env = OrtEnvironment.getEnvironment();
    val session =
      env.createSession(file.getAbsolutePath, new OrtSession.SessionOptions())

    def stenToORT(t: STen) = {
      t.scalarTypeByte match {
        case 4 =>
          val buffer = LongBuffer.wrap(t.toLongArray)
          OnnxTensor.createTensor(env, buffer, t.shape.toArray)
        case 6 =>
          val buffer = FloatBuffer.wrap(t.toFloatArray)
          OnnxTensor.createTensor(env, buffer, t.shape.toArray)
        case 7 =>
          val buffer = DoubleBuffer.wrap(t.toDoubleArray)
          OnnxTensor.createTensor(env, buffer, t.shape.toArray)
      }
    }
    val result = session
      .run(
        input.map { case (i, v) => (i, stenToORT(v)) }.asJava
      )
      .get(0)
      .asInstanceOf[OnnxTensor]

    `type` match {
      case 4 =>
        val buffer = result.getLongBuffer()
        val dim = result.getInfo.getShape.toList
        val ar = Array.ofDim[Long](dim.foldLeft(1L)(_ * _).toInt)
        STen.fromLongArray(buffer.get(ar).array, dim, CPU)
      case 6 =>
        val buffer = result.getFloatBuffer()
        val dim = result.getInfo.getShape.toList
        val ar = Array.ofDim[Float](dim.foldLeft(1L)(_ * _).toInt)
        STen.fromFloatArray(buffer.get(ar).array, dim, CPU)
      case 7 =>
        val buffer = result.getDoubleBuffer()
        val dim = result.getInfo.getShape.toList
        val ar = Array.ofDim[Double](dim.foldLeft(1L)(_ * _).toInt)
        STen.fromDoubleArray(buffer.get(ar).array, dim, CPU, DoublePrecision)
    }

  }

  def testBinaryOps(
      name: String,
      shape1: Seq[Long],
      shape2: Seq[Long],
      op: Scope => (Variable, Variable) => Variable,
      expectNoImplemenation: Boolean = false,
      tOpt: STenOptions = STenOptions.f
  ) =
    test(
      name + (if (expectNoImplemenation) " !!! No runtime implementation !!!"
              else "")
    ) {
      Scope.root { implicit scope =>
        val tt1 = STen.ones(shape1, tOpt)
        val tt2 = STen.ones(shape2, tOpt) * 3
        val t1 = const(tt1)
        val t2 = param(tt2)
        val output = op(scope)(t1, t2)
        val file = java.io.File.createTempFile("tmp", ".onnx")
        serializeToFile(
          file,
          output,
          modelDocString = "model description"
        ) {
          case x if x == t1 =>
            VariableInfo(
              variable = t1,
              input = true,
              name = "t1",
              docString = "baa"
            )
          case x if x == t2 =>
            VariableInfo(variable = t2, input = false, name = "t2")
          case x if x == output =>
            VariableInfo(
              variable = output,
              input = false,
              name = "output",
              docString = "boo"
            )
        }
        // println(file)

        try {
          val result =
            runModel(file, Map("t1" -> tt1), output.value.scalarTypeByte)
          assert(result.shape == output.shape)
          assert((result * 10000).round.equalDeep((output.value * 10000).round))
        } catch {
          case e: ai.onnxruntime.OrtException
              if expectNoImplemenation && e.getMessage
                .contains("ORT_NOT_IMPLEMENTED") =>
        }
      }
    }

  testBinaryOps("Mul", List(3, 3), List(3, 3), implicit scope => _ * _)
  testBinaryOps(
    "Mul Const",
    List(3, 3),
    List(3, 3),
    implicit scope => _ * _ * 3d
  )
  testBinaryOps(
    "View",
    List(3, 3),
    List(3, 3),
    implicit scope => (a, b) => a.view(b.shape)
  )
  testBinaryOps(
    "Cat",
    List(3, 3),
    List(3, 3),
    implicit scope => (a, b) => a.cat(b, 0)
  )
  testBinaryOps(
    "MatMul",
    List(3, 2),
    List(2, 3),
    implicit scope => (a, b) => a.mm(b)
  )
  testBinaryOps(
    "Transpose",
    List(3, 3),
    List(3, 3),
    implicit scope => (a, b) => a.cat(b, 0).t
  )
  testBinaryOps(
    "Add",
    List(3, 3),
    List(3, 3),
    implicit scope => (a, b) => a + b
  )
  testBinaryOps(
    "Add Cosnt",
    List(3, 3),
    List(3, 3),
    implicit scope => (a, b) => a + b + 3d
  )
  testBinaryOps(
    "Sub",
    List(3, 3),
    List(3, 3),
    implicit scope => (a, b) => a - b
  )
  testBinaryOps(
    "Div",
    List(3, 3),
    List(3, 3),
    implicit scope => (a, b) => a / b
  )
  testBinaryOps(
    "Exp",
    List(3, 3),
    List(3, 3),
    implicit scope => (a, b) => a.cat(b, 0).exp
  )
  testBinaryOps(
    "Sin",
    List(3, 3),
    List(3, 3),
    implicit scope => (a, b) => a.cat(b, 0).sin
  )
  testBinaryOps(
    "Tanh",
    List(3, 3),
    List(3, 3),
    implicit scope => (a, b) => a.cat(b, 0).tanh
  )
  testBinaryOps(
    "Cos",
    List(3, 3),
    List(3, 3),
    implicit scope => (a, b) => a.cat(b, 0).cos
  )
  testBinaryOps(
    "Tan",
    List(3, 3),
    List(3, 3),
    implicit scope => (a, b) => a.cat(b, 0).tan
  )
  testBinaryOps(
    "ATan",
    List(3, 3),
    List(3, 3),
    implicit scope => (a, b) => a.cat(b, 0).atan
  )
  testBinaryOps(
    "relu",
    List(3, 3),
    List(3, 3),
    implicit scope => (a, b) => a.cat(b, 0).relu
  )
  testBinaryOps(
    "gelu",
    List(3, 3),
    List(3, 3),
    implicit scope => (a, b) => a.cat(b, 0).gelu
  )
  testBinaryOps(
    "Pow Const",
    List(3, 3),
    List(3, 3),
    implicit scope => (a, b) => (a * b).pow(3d)
  )
  testBinaryOps(
    "Pow",
    List(3, 3),
    List(1),
    implicit scope => (a, b) => a.pow(b)
  )
  testBinaryOps(
    "OneHot",
    List(3),
    List(1),
    implicit scope => (a, _) => a.oneHot(2),
    tOpt = STenOptions.l
  )
  testBinaryOps(
    "sum 1",
    List(3, 3),
    List(1),
    implicit scope => (a, _) => a.sum(dim = List(0), keepDim = false)
  )
  testBinaryOps(
    "sum 2",
    List(3, 3),
    List(1),
    implicit scope => (a, _) => a.sum(dim = List(0), keepDim = true)
  )
  testBinaryOps(
    "sum 3",
    List(3, 3),
    List(1),
    implicit scope => (a, _) => a.mean(dim = List(), keepDim = false)
  )
  testBinaryOps(
    "mean 1",
    List(3, 3),
    List(1),
    implicit scope => (a, _) => a.mean(dim = List(0), keepDim = false)
  )
  testBinaryOps(
    "mean 2",
    List(3, 3),
    List(1),
    implicit scope => (a, _) => a.mean(dim = List(0), keepDim = true)
  )
  testBinaryOps(
    "mean 3",
    List(3, 3),
    List(1),
    implicit scope => (a, _) => a.sum(dim = List(), keepDim = false)
  )
  testBinaryOps(
    "expand as",
    List(1),
    List(3, 3),
    implicit scope => (a, b) => a.expandAs(b.value)
  )
  testBinaryOps(
    "bmm",
    List(3, 2, 3),
    List(3, 3, 2),
    implicit scope => (a, b) => a.bmm(b)
  )
  testBinaryOps(
    "logsoftmax",
    List(2, 3),
    List(1),
    implicit scope => (a, _) => a.logSoftMax(1)
  )
  testBinaryOps(
    "sigmoid",
    List(2, 3),
    List(1),
    implicit scope => (a, _) => a.sigmoid
  )
  testBinaryOps(
    "dropout",
    List(2, 3),
    List(1),
    implicit scope => (a, _) => a.dropout(prob = 0.5, train = false)
  )
  testBinaryOps(
    "flatten",
    List(2, 3, 3, 4),
    List(1),
    implicit scope => (a, _) => a.flatten(1)
  )
  testBinaryOps(
    "conv1d",
    List(1, 1, 3),
    List(1, 1, 3),
    implicit scope =>
      (a, b) => {
        val bb = const(STen.zeros(List(1), STenOptions.f))
        new Conv1D(scope, a, b, bb, 1, 1, 1, 1).value
      }
  )
  testBinaryOps(
    "conv2d",
    List(1, 1, 3, 3),
    List(1, 1, 3, 3),
    implicit scope =>
      (a, b) => {
        val bb = const(STen.zeros(List(1), STenOptions.f))
        new Conv2D(scope, a, b, bb, 1, 1, 1, 1).value
      }
  )
  testBinaryOps(
    "maxpool1d",
    List(1, 1, 5),
    List(1),
    implicit scope =>
      (a, _) => {
        new MaxPool1D(scope, a, 3, 1, 1).value
      }
  )
  testBinaryOps(
    "maxpool2d",
    List(1, 1, 5, 5),
    List(1),
    implicit scope =>
      (a, _) => {
        new MaxPool2D(scope, a, 3, 1, 1, 1).value
      }
  )
  testBinaryOps(
    "avgpool2d",
    List(1, 1, 5, 5),
    List(1),
    implicit scope =>
      (a, _) => {
        new AvgPool2D(scope, a, 3, 1, 1).value
      }
  )
  testBinaryOps(
    "batchnorm1d",
    List(1, 5),
    List(5),
    implicit scope =>
      (a, b) => {
        val bias = const(STen.zeros(List(5), STenOptions.f))
        val m = STen.zeros(List(5), STenOptions.f)
        val v = STen.zeros(List(5), STenOptions.f)
        new BatchNorm(scope, a, b, bias, m, v, false, 0.99, 1e-5).value
      }
  )
  testBinaryOps(
    "batchnorm2d",
    List(1, 3, 5),
    List(3),
    implicit scope =>
      (a, b) => {
        val bias = const(STen.zeros(List(3), STenOptions.f))
        val m = STen.zeros(List(3), STenOptions.f)
        val v = STen.zeros(List(3), STenOptions.f)
        new BatchNorm2D(scope, a, b, bias, m, v, false, 0.99, 1e-5).value
      }
  )

  testBinaryOps(
    "assign",
    List(2, 3),
    List(2, 3),
    implicit scope => (a, b) => a.assign(b * 3)
  )
  testBinaryOps(
    "cast",
    List(2, 3),
    List(2, 3),
    implicit scope => (a, _) => a.cast(DoublePrecision)
  )
  testBinaryOps(
    "select",
    List(2, 3),
    List(2, 3),
    implicit scope => (a, _) => a.select(0, 1)
  )
  testBinaryOps(
    "indexSelect",
    List(2, 3),
    List(2, 3),
    implicit scope =>
      (a, _) => {
        val i = const(STen.zeros(List(5), STenOptions.l))
        a.indexSelect(0, i)
      }
  )
  testBinaryOps(
    "stack",
    List(2, 3),
    List(2, 3),
    implicit scope => (a, b) => Variable.stack(List(a, b), 1)
  )

}
