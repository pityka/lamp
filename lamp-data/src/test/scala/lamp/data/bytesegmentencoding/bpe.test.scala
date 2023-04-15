package lamp.data.bytesegmentencoding

import org.scalatest.funsuite.AnyFunSuite

class ByteSegmentEncodingSuite extends AnyFunSuite {
  test("byte segment encoding") {
    val corpus =
      Array.apply[Byte]('0', '0', '1', '2',',', '0', '1', '2', '1', '4', '2', '5', '2', '6', '9', '9', '9').map(_.toByte)
    val trained : Vector[(Vector[Byte],Char)] =
      lamp.data.bytesegmentencoding.train(corpus, 0.toChar, 100.toChar, 3)
    val encoded =
      lamp.data.bytesegmentencoding.encode(corpus, trained, 0.toChar)
    val decoded =
      lamp.data.bytesegmentencoding.decode(encoded, trained, '?'.toByte)
    assert(corpus.toVector == decoded.toVector)
  }
}
