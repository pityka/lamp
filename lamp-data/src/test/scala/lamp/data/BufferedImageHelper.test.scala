package lamp.data

import org.scalatest.funsuite.AnyFunSuite
import org.saddle._
import java.awt.image.BufferedImage
import lamp.util.NDArray
import java.awt.Color

class BufferedImageHelperSuite extends AnyFunSuite {
  test("to tensor") {
    val bi = new BufferedImage(3, 4, BufferedImage.TYPE_INT_ARGB)
    0 until 3 foreach { i =>
      0 until 4 foreach { j => bi.setRGB(i, j, new Color(0, 0, 0).getRGB()) }
    }
    bi.setRGB(0, 0, new Color(255, 0, 0).getRGB())
    bi.setRGB(2, 3, new Color(0, 255, 0).getRGB())
    val biData = bi.getRGB(0, 0, 3, 4, null, 0, 3).toVector
    val tensor = BufferedImageHelper.toFloatTensor(bi)
    val expected = Vec(255.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 255.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0).map(_.toFloat)
    assert(NDArray.tensorToFloatNDArray(tensor).toVec == expected)
    val bi2 = BufferedImageHelper.fromFloatTensor(tensor)
    val bi2Data = bi2.getRGB(0, 0, 3, 4, null, 0, 3).toVector
    assert(bi2Data == biData)
  }
}
