package lamp.data

import org.scalatest.funsuite.AnyFunSuite
import lamp.nn._
import lamp.util.NDArray
import java.io.File
import lamp.CPU
import lamp.Scope
import lamp.STenOptions
import cats.effect.unsafe.implicits.global

class ReadWriteSuite extends AnyFunSuite {

  test("checkpoint modules - float") {
    Scope.root { implicit scope =>
      val topt = STenOptions.f
      val net = Sequential(Linear(5, 5, topt), Linear(5, 5, topt))
      val file = File.createTempFile("prefix", "suffx")
      file.delete
      Writer.writeCheckpoint(file, net).unsafeRunSync()

      val net2 = Sequential(Linear(5, 5, topt), Linear(5, 5, topt))
      Reader.loadFromFile(net2, file, CPU).unsafeRunSync()
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
      Reader.loadFromFile(net2, file, CPU).unsafeRunSync()
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
