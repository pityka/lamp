package lamp.data.safetensors

import org.scalatest.funsuite.AnyFunSuite
import lamp._
import java.io.File
import lamp.data.safetensors.SafeTensorReader

class SafeTensorsSuite extends AnyFunSuite {
  test("read") {
    Scope.root { implicit scope =>
      val tensorList = SafeTensorReader.read(
        new File("../test_data/model.safetensors"),
        CPU
      )
      assert(tensorList.tensors.size == 132)
      assert(
        tensorList
          .tensors("decoder.block.4.layer.2.layer_norm.weight")
          .toFloatArray
          .length == 512
      )
      ()
    }
  }
}
