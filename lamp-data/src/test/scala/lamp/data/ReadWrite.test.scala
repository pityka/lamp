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
import java.io.File
import lamp.CPU

class ReadWriteSuite extends AnyFunSuite {
  test("to tensor") {
    val t = ATen.ones(Array(3, 3, 3), TensorOptions.dtypeFloat())
    val t2 = Reader
      .readTensorFromArray[Float](
        Writer.writeTensorIntoArray[Float](t).right.get,
        CPU
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
      List(tf, td),
      channel
    )
    channel.close()
    assert(success.isRight)
    val is = new java.io.FileInputStream(f)
    val channelIn = is.getChannel()
    val read = Reader
      .readTensorsFromChannel(
        List(ScalarTagFloat, ScalarTagDouble),
        channelIn,
        CPU
      )
      .right
      .get

    assert(
      NDArray.tensorToFloatNDArray(read(0)).toVec.length == 27
    )
    assert(
      NDArray.tensorToFloatNDArray(read(0)).toVec == vec.ones(27).map(_.toFloat)
    )
    assert(
      NDArray.tensorToNDArray(read(1)).toVec.length == 27
    )
    assert(
      NDArray.tensorToNDArray(read(1)).toVec == vec.ones(27)
    )
  }
  test("checkpoint modules - float") {
    val topt = TensorOptions.dtypeFloat()
    val net = Sequential(Linear(5, 5, topt), Linear(5, 5, topt))
    val file = File.createTempFile("prefix", "suffx")
    file.delete
    Writer.writeCheckpoint(file, net).unsafeRunSync()
    val net2 = Sequential(Linear(5, 5, topt), Linear(5, 5, topt))
    val loaded = Reader.loadFromFile(net2, file, CPU).unsafeRunSync().right.get
    loaded.state.zip(net.state).foreach {
      case ((loaded, _), (orig, _)) =>
        val ndL = NDArray.tensorToFloatNDArray(loaded.value)
        val ndO = NDArray.tensorToFloatNDArray(orig.value)
        assert(ndL.toVec == ndO.toVec)
    }

  }
  test("checkpoint modules - mixed") {
    val topt = TensorOptions.dtypeFloat()
    val net = Sequential(Linear(5, 5, topt), Linear(5, 5, topt.toDouble))
    val file = File.createTempFile("prefix", "suffx")
    file.delete
    Writer.writeCheckpoint(file, net).unsafeRunSync()
    val net2 = Sequential(Linear(5, 5, topt), Linear(5, 5, topt.toDouble()))
    val loaded = Reader.loadFromFile(net2, file, CPU).unsafeRunSync().right.get
    loaded.state.zip(net.state).foreach {
      case ((loaded, _), (orig, _)) =>
        loaded.options.scalarTypeByte() match {
          case 6 =>
            val ndL = NDArray.tensorToFloatNDArray(loaded.value)
            val ndO = NDArray.tensorToFloatNDArray(orig.value)
            assert(ndL.toVec == ndO.toVec)
          case 7 =>
            val ndL = NDArray.tensorToNDArray(loaded.value)
            val ndO = NDArray.tensorToNDArray(orig.value)
            assert(ndL.toVec == ndO.toVec)
        }
    }

  }
}
