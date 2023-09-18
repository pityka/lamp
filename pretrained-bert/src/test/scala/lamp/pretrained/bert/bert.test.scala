package lamp.pretrained.bert

import org.scalatest.funsuite.AnyFunSuite
import lamp._
import java.io.File
import lamp.data.safetensors.SafeTensorReader

class SafeTensorsSuite extends AnyFunSuite {
  test("read") {
    Scope.root { implicit scope =>
     
      val bertFile = new File("../test_data/bert.model.safetensors"),
     
      ()
    }
  }
}
