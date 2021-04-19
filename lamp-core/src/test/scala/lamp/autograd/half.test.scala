package lamp.autograd

import org.scalatest.funsuite.AnyFunSuite
import lamp.Scope
import lamp.STen
import lamp.CudaDevice
import lamp.SinglePrecision
import lamp.CPU
import lamp.Device

class HalfPrecisionSuite extends AnyFunSuite {
  def doIt(device: Device) = {
    Scope.root { implicit scope =>
      val v1 = const(STen.rand(List(512, 512), device.options(SinglePrecision)))

      { // warmup
        implicit val conf = lamp.autograd.implicits.defaultGraphConfiguration
        (0 until 10).foreach { _ =>
          v1.mm(v1)
        }
      }

      val t1 = System.nanoTime

      {
        implicit val conf = lamp.autograd.implicits.defaultGraphConfiguration
        (0 until 100).foreach { _ =>
          v1.mm(v1)
        }
      }
      val duration1 = System.nanoTime - t1
      println(s"Duration float,$device (sec): " + (duration1 * 1e-9))
      val t2 = System.nanoTime

      if (device != CPU) {
        {
          implicit val conf = lamp.autograd.implicits.defaultGraphConfiguration
            .copy(downCastEnabled = true)
          (0 until 100).foreach { _ =>
            v1.mm(v1)
          }
        }
        val duration2 = System.nanoTime - t2
        println(s"Duration amp,$device (sec): " + (duration2 * 1e-9))
      }

      println("\n\n")

    }
  }

  test("half precision benchmark cuda, cpu") {
    doIt(CPU)
    if (aten.Tensor.cudnnAvailable()) {
      doIt(CudaDevice(0))
    }
  }
}
