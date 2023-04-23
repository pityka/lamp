package lamp.data.bytesegmentencoding

import org.scalatest.funsuite.AnyFunSuite

class ByteSegmentEncodingSuite extends AnyFunSuite {
  test("byte segment encoding") {
    val corpus =
      Array.apply[Byte]('0', '0', '1', '2', '0', '1', '2', '1', '4', '2', '5',
        '2', '6', '9', '9', '9')
    val trained =
      lamp.data.bytesegmentencoding.train(corpus, 0.toChar, 100.toChar, 3)
    val encoded =
      lamp.data.bytesegmentencoding.encode(corpus, trained, 0.toChar)
    val decoded =
      lamp.data.bytesegmentencoding.decode(encoded, trained, '?'.toByte)
    assert(corpus.toVector == decoded.toVector)
    assert(
      trained.map(v => (v._1.map(_.toChar).mkString -> v._2.toInt)) == Vector(
        ("2", 0),
        ("9", 1),
        ("6", 2),
        ("1", 3),
        ("4", 4),
        ("5", 5),
        ("0", 6),
        ("012", 7),
        ("99", 8),
        ("12", 9),
        ("01", 10),
        ("201", 11),
        ("25", 12),
        ("269", 13),
        ("121", 14),
        ("999", 15),
        ("142", 16),
        ("21", 17),
        ("252", 18),
        ("14", 19),
        ("52", 20),
        ("00", 21),
        ("425", 22),
        ("26", 23),
        ("120", 24),
        ("699", 25),
        ("526", 26),
        ("20", 27),
        ("214", 28),
        ("42", 29),
        ("69", 30),
        ("001", 31)
      )
    )
  }
}
