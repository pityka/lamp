package lamp.tabular

import org.saddle._
import org.saddle.ops.BinOps._
import org.scalatest.funsuite.AnyFunSuite
import aten.ATen
import lamp.autograd._
import aten.TensorOptions
import org.scalatest.Tag
import lamp.syntax
import lamp.util.NDArray
import aten.Tensor
import cats.effect.IO
import cats.effect.concurrent.Ref
import lamp.nn._
import lamp.SinglePrecision
import lamp.CPU
import lamp.CudaDevice
import lamp.Device
import lamp.DoublePrecision
import scribe.Logger

class ECDFSuite extends AnyFunSuite {
  test("ecdf") {
    val values = Vec(3d, 1d, 1d, 2d, 2d, 4d)
    val ecdf = ECDF(values)
    assert(ecdf.x == Vec(1d, 1d, 2d, 2d, 3d, 4d))
    assert(ecdf.y == Vec(2d / 6, 2d / 6, 4d / 6, 4d / 6, 5d / 6, 6d / 6))
    assert(ecdf(4d) == 1d)
    assert(ecdf(5d) == 1d)
    assert(ecdf(0d) == 0d)
    assert(ecdf(1d) == 2d / 6)
    assert(ecdf(2d) == 4d / 6)
    assert(ecdf(3d) == 5d / 6)
    assert(ecdf(4d) == 1d)
    assert(ecdf(3.5d) == 5d / 6)
    assert(ecdf(0.5d) == 0d)

    assert(ecdf.inverse(-1d) == 1d)
    assert(ecdf.inverse(0d) == 1d)
    assert(ecdf.inverse(1d) == 4d)
    assert(ecdf.inverse(2d / 6) == 1d)
    assert(ecdf.inverse(4d / 6) == 2d)
    assert(ecdf.inverse(5d / 6) == 3d)
    assert(ecdf.inverse(4.5 / 6) == 2.5d)
    assert(ecdf.inverse(5.5 / 6) == 3.4999999999999996)
    assert(ecdf.inverse(3d / 6) == 1.5)
    assert(ecdf.inverse(1.1d) == 4d)
  }
}
