package lamp.data

import org.scalatest.funsuite.AnyFunSuite
import org.saddle._
import lamp.nn._
import aten.ATen
import lamp.util.NDArray
import org.saddle.scalar.ScalarTagFloat
import org.saddle.scalar.ScalarTagDouble
import java.io.File
import lamp.CPU
import lamp.Scope
import lamp.STenOptions
import cats.effect.unsafe.implicits.global
import lamp.autograd.implicits.defaultGraphConfiguration

class ReadWriteSuite extends AnyFunSuite {
  test("to tensor") {
    val t = ATen.ones(Array(3, 3, 3), STenOptions.f.value)
    val t2 = Reader
      .readTensorFromArray[Float](
        Writer.writeTensorIntoArray[Float](t).toOption.get,
        CPU
      )
    assert(
      NDArray.tensorToFloatNDArray(t).toVec == NDArray
        .tensorToFloatNDArray(t2.toOption.get)
        .toVec
    )
  }
  test("tensors") {
    val tf = ATen.ones(Array(3, 3, 3), STenOptions.f.value)
    val td = ATen.ones(Array(3, 3, 3), STenOptions.d.value)
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
      .toOption
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
    Scope.root { implicit scope =>
      val topt = STenOptions.f
      val net = Sequential(Linear(5, 5, topt), Linear(5, 5, topt))
      val file = File.createTempFile("prefix", "suffx")
      file.delete
      Writer.writeCheckpoint(file, net).unsafeRunSync()
      val net2 = Sequential(Linear(5, 5, topt), Linear(5, 5, topt))
      Reader.loadFromFile(net2, file, CPU).unsafeRunSync().toOption.get
      net2.state.zip(net.state).foreach { case ((loaded, _), (orig, _)) =>
        val ndL = NDArray.tensorToFloatNDArray(loaded.value.value)
        val ndO = NDArray.tensorToFloatNDArray(orig.value.value)
        assert(ndL.toVec == ndO.toVec)
      }
    }
  }
  test("checkpoint modules - mixed") {
    Scope.root { implicit scope =>
      val topt = STenOptions.f
      val net = Sequential(Linear(5, 5, topt), Linear(5, 5, topt.toDouble))
      val file = File.createTempFile("prefix", "suffx")
      file.delete
      Writer.writeCheckpoint(file, net).unsafeRunSync()
      val net2 = Sequential(Linear(5, 5, topt), Linear(5, 5, topt.toDouble))
      Reader.loadFromFile(net2, file, CPU).unsafeRunSync().toOption.get
      net2.state.zip(net.state).foreach { case ((loaded, _), (orig, _)) =>
        loaded.value.scalarTypeByte match {
          case 6 =>
            val ndL = NDArray.tensorToFloatNDArray(loaded.value.value)
            val ndO = NDArray.tensorToFloatNDArray(orig.value.value)
            assert(ndL.toVec == ndO.toVec)
          case 7 =>
            val ndL = NDArray.tensorToNDArray(loaded.value.value)
            val ndO = NDArray.tensorToNDArray(orig.value.value)
            assert(ndL.toVec == ndO.toVec)
        }
      }

    }
  }
}
