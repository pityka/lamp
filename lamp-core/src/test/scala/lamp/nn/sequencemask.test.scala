package lamp.nn

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

class SequenceMaskSuite extends AnyFunSuite {
  implicit val pool = new AllocatedVariablePool
  test("sequence mask") {
    val nd3x2L = NDArray(Array(0L, 1L, 0L, 1L, 0L, 0L), List(3, 2))
    val nd3x2x2 = NDArray(vec.ones(24).toArray, List(3, 2, 4))
    val tokens = NDArray.tensorFromLongNDArray(nd3x2L, false)
    val maskable = NDArray.tensorFromNDArray(nd3x2x2, false)
    val masked = SequenceMask.apply(
      tokens = const(tokens),
      maskable = const(maskable),
      maskedToken = 1L,
      fill = -1d
    )
    val maskedND = NDArray.tensorToNDArray(masked.value)
    val expected = NDArray(
      Array(1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0,
        -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
      List(3, 2, 4)
    )
    assert(maskedND.toVec == expected.toVec)
  }

}
