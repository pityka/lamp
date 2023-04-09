package lamp.data.bytesegmentencoding

import org.scalatest.funsuite.AnyFunSuite

class ByteSegmentEncodingSuite extends AnyFunSuite {
  test("byte segment encoding") {
    val corpus =
      Array.apply[Byte](0, 0, 1, 2, 0, 1, 2, 1, 4, 2, 5, 2, 6, 9, 9, 9)
    val trained =
      lamp.data.bytesegmentencoding.train(corpus, 0.toChar, 100.toChar, 3)
    val encoded = lamp.data.bytesegmentencoding.encode(corpus, trained, 0.toChar)
    val decoded = lamp.data.bytesegmentencoding.decode(encoded, trained, '?'.toByte)
    assert(corpus.toVector == decoded.toVector)
    assert(
      trained == Vector(
        (Vector(0, 1, 2), 0),
        (Vector(9, 9), 1),
        (Vector(1, 2), 2),
        (Vector(0, 1), 3),
        (Vector(2), 4),
        (Vector(0), 5),
        (Vector(0, 0, 1), 6),
        (Vector(9, 9, 9), 7),
        (Vector(1, 2, 0), 8),
        (Vector(1, 2, 1), 9),
        (Vector(4, 2, 5), 10),
        (Vector(2, 6, 9), 11),
        (Vector(2, 1, 4), 12),
        (Vector(1), 13),
        (Vector(2, 5, 2), 14),
        (Vector(6, 9, 9), 15),
        (Vector(2, 0, 1), 16),
        (Vector(1, 4, 2), 17),
        (Vector(5, 2, 6), 18),
        (Vector(9), 19),
        (Vector(2, 0), 20),
        (Vector(4, 2), 21),
        (Vector(2, 5), 22),
        (Vector(5, 2), 23),
        (Vector(2, 1), 24),
        (Vector(2, 6), 25),
        (Vector(6, 9), 26),
        (Vector(0, 0), 27),
        (Vector(1, 4), 28),
        (Vector(5), 29),
        (Vector(4), 30),
        (Vector(6), 31)
      )
    )
  }
}
