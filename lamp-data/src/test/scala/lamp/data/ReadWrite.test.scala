package lamp.data

import org.scalatest.funsuite.AnyFunSuite
import org.saddle._
import org.saddle.order._
import org.saddle.ops.BinOps._
import lamp.autograd.{Variable, const, param, TensorHelpers}
import lamp.nn._
import aten.ATen
import aten.TensorOptions
import java.awt.image.BufferedImage
import lamp.util.NDArray
import java.awt.Color
import org.saddle.scalar.ScalarTagFloat
import org.saddle.scalar.ScalarTagDouble

class ReadWriteSuite extends AnyFunSuite {
  test("to tensor") {
    val t = ATen.ones(Array(3, 3, 3), TensorOptions.dtypeFloat())
    val t2 = Reader
      .readTensorFromArray[Float](
        Writer.writeTensorIntoArray[Float](t).right.get
      )
    assert(
      NDArray.tensorToFloatNDArray(t).toVec == NDArray
        .tensorToFloatNDArray(t2.right.get)
        .toVec
    )
  }
  test("tensors") {
    val tf = ATen.ones(Array(3, 3, 3), TensorOptions.dtypeFloat())
    val td = ATen.ones(Array(3, 3, 3), TensorOptions.dtypeDouble())
    val f = java.io.File.createTempFile("temp", "")
    val os = new java.io.FileOutputStream(f)
    val channel = os.getChannel()
    val success = Writer.writeTensorsIntoChannel(
      List(ScalarTagFloat -> tf, ScalarTagDouble -> td),
      channel
    )
    channel.close()
    assert(success.isRight)
    val is = new java.io.FileInputStream(f)
    val channelIn = is.getChannel()
    val read = Reader
      .readTensorsFromChannel(List(ScalarTagFloat, ScalarTagDouble), channelIn)
      .right
      .get

    assert(
      NDArray.tensorToFloatNDArray(read(0)).toVec.length == 27
    )
    assert(
      NDArray.tensorToNDArray(read(1)).toVec.length == 27
    )
  }
}
