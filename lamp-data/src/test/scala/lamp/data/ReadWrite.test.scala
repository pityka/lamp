package lamp.data

import org.scalatest.funsuite.AnyFunSuite
import lamp.nn._
import lamp.util.NDArray
import java.io.File
import lamp.CPU
import lamp.Scope
import lamp.STenOptions
import cats.effect.unsafe.implicits.global
import lamp.STen
import lamp.autograd.NDArraySyntax._ 

class ReadWriteSuite extends AnyFunSuite {

  test("io empty") {
    Scope.root { implicit scope =>
      val file = File.createTempFile("prefix", "suffx")
      val st1 = STen.ones(List(3, 3))
      val st2 = STen.zeros(List(0))
      val st3 = STen.zeros(List(0, 0), STenOptions.f)
      val st4 = STen.zeros(List(0), STenOptions.l)
      val st5 = STen.ones(List(3, 3))
      val st6 = STen.ones(List(3), STenOptions.b)
      val list = List(st1, st2, st3, st4, st5,st6)
      assert(
        Writer
          .writeTensorsIntoFile(list, file)
          .unsafeRunSync()
          .isRight
      )
      val read1 = Reader.readTensorsFromFile(file, lamp.CPU, false)
      val read2 = Reader.readTensorsFromFile(file, lamp.CPU, true)
      list.zip(read1).foreach(a => assert(a._1.equalDeep(a._2)))
      list.zip(read2).foreach(a => assert(a._1.equalDeep(a._2)))
    }
  }
  test("io empty 2") {
    Scope.root { implicit scope =>
      val file = File.createTempFile("prefix", "suffx")
      val st2 = STen.zeros(List(0))
      val st3 = STen.zeros(List(0, 0), STenOptions.f)
      val st4 = STen.zeros(List(0), STenOptions.l)
      val list = List(st2, st3, st4)
      assert(
        Writer
          .writeTensorsIntoFile(list, file)
          .unsafeRunSync()
          .isRight
      )
      val read1 = Reader.readTensorsFromFile(file, lamp.CPU, false)
      val read2 = Reader.readTensorsFromFile(file, lamp.CPU, true)
      list.zip(read1).foreach(a => assert(a._1.equalDeep(a._2)))
      list.zip(read2).foreach(a => assert(a._1.equalDeep(a._2)))
    }
  }

  test("checkpoint modules - float") {
    Scope.root { implicit scope =>
      val topt = STenOptions.f
      val net = Sequential(Linear(5, 5, topt), Linear(5, 5, topt))
      val file = File.createTempFile("prefix", "suffx")
      file.delete
      Writer.writeCheckpoint(file, net).unsafeRunSync()

      val net2 = Sequential(Linear(5, 5, topt), Linear(5, 5, topt))
      Reader.loadFromFile(net2, file, CPU, false).unsafeRunSync()
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
      Reader.loadFromFile(net2, file, CPU, false).unsafeRunSync()
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
