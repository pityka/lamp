package lamp.data.bytesegmentencoding

import org.scalatest.funsuite.AnyFunSuite

class ByteSegmentEncodingSuite extends AnyFunSuite {
  test("byte segment encoding") {
    val corpus =
      Array.apply[Byte](0, 0, 1, 2, 0, 1, 2, 1, 4, 2, 5, 2, 6, 9, 9, 9)
    val trained =
      lamp.data.bytesegmentencoding.train(corpus, 0.toChar, 100.toChar, 3)
    val encoded =
      lamp.data.bytesegmentencoding.encode(corpus, trained, 0.toChar)
    val decoded =
      lamp.data.bytesegmentencoding.decode(encoded, trained, '?'.toByte)
    assert(corpus.toVector == decoded.toVector)
    assert(
      trained == Vector(
        (Vector(5), 0),
        (Vector(0), 1),
        (Vector(4), 2),
        (Vector(6), 3),
        (Vector(1), 4),
        (Vector(2), 5),
        (Vector(9), 6),
        (Vector(0, 1, 2), 7),
        (Vector(9, 9), 8),
        (Vector(1, 2), 9),
        (Vector(0, 1), 10),
        (Vector(2, 0), 11),
        (Vector(0, 0, 1), 12),
        (Vector(4, 2), 13),
        (Vector(9, 9, 9), 14),
        (Vector(1, 2, 0), 15),
        (Vector(2, 5), 16),
        (Vector(1, 2, 1), 17),
        (Vector(4, 2, 5), 18),
        (Vector(2, 6, 9), 19),
        (Vector(2, 1, 4), 20),
        (Vector(5, 2), 21),
        (Vector(2, 1), 22),
        (Vector(2, 6), 23),
        (Vector(2, 5, 2), 24),
        (Vector(6, 9, 9), 25),
        (Vector(2, 0, 1), 26),
        (Vector(1, 4, 2), 27),
        (Vector(6, 9), 28),
        (Vector(0, 0), 29),
        (Vector(5, 2, 6), 30),
        (Vector(1, 4), 31)
      )
    )
  }
}
