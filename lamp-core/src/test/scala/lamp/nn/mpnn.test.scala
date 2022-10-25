package lamp.nn

import org.scalatest.funsuite.AnyFunSuite

import lamp.Scope
import lamp.STen
import lamp.autograd._
import lamp.nn.graph.MPNN
import org.scalatest.compatible.Assertion

class MPNNSuite extends AnyFunSuite {
implicit val AssertionIsMovable : lamp.EmptyMovable[Assertion] = lamp.Movable.empty[Assertion]
  test("count occurrences") {
    Scope.root { implicit scope =>
      val t = STen.fromLongArray(Array(1L, 1L, 2L, 1L, 3L, 2L, 1L))
      val r = graph.MPNN.countOccurences(t, 5)
      assert(r.toLongArray.toVector == Vector(0, 4, 2, 1, 0))
    }
  }
  test("aggregate") {
    Scope.root { implicit scope =>
      val message = const(STen.ones(List(2, 3)))
      val edgeI = STen.fromLongArray(Array(0L, 0L))
      val edgeJ = STen.fromLongArray(Array(1L, 2L))

      import lamp.saddle._
      import org.saddle._

      assert(
        MPNN
          .aggregate(
            3,
            message,
            edgeI,
            edgeJ,
            degreeNormalizeI = false,
            degreeNormalizeJ = false,
            aggregateJ = false
          )
          .value
          .toMat == Mat(Vec(0d, 1d, 1d), Vec(0d, 1d, 1d), Vec(0d, 1d, 1d))
      )
      assert(
        MPNN
          .aggregate(
            3,
            message,
            edgeI,
            edgeJ,
            degreeNormalizeI = false,
            degreeNormalizeJ = false,
            aggregateJ = true
          )
          .value
          .toMat == Mat(Vec(2d, 1d, 1d), Vec(2d, 1d, 1d), Vec(2d, 1d, 1d))
      )
      assert(
        MPNN
          .aggregate(
            3,
            message,
            edgeI,
            edgeJ,
            degreeNormalizeI = true,
            degreeNormalizeJ = false,
            aggregateJ = false
          )
          .value
          .toMat == Mat(Vec(0d, 0.5, 0.5), Vec(0d, 0.5, 0.5), Vec(0d, 0.5, 0.5))
      )
      assert(
        MPNN
          .aggregate(
            3,
            message,
            edgeI,
            edgeJ,
            degreeNormalizeI = false,
            degreeNormalizeJ = true,
            aggregateJ = false
          )
          .value
          .toMat == Mat(Vec(0d, 1d, 1d), Vec(0d, 1d, 1d), Vec(0d, 1d, 1d))
      )

      assert(
        MPNN
          .aggregate(
            3,
            message,
            edgeI,
            edgeJ,
            degreeNormalizeI = true,
            degreeNormalizeJ = false,
            aggregateJ = true
          )
          .value
          .toMat == Mat(
          Vec(1d, 0.5, 0.5d),
          Vec(1d, 0.5, 0.5d),
          Vec(1d, 0.5, 0.5d)
        )
      )
      assert(
        MPNN
          .aggregate(
            3,
            message,
            edgeI,
            edgeJ,
            degreeNormalizeI = false,
            degreeNormalizeJ = true,
            aggregateJ = true
          )
          .value
          .toMat == Mat(
          Vec(2d, 1d, 1d),
          Vec(2d, 1d, 1d),
          Vec(2d, 1d, 1d)
        )
      )

      assert(
        MPNN
          .aggregate(
            3,
            message,
            edgeI,
            edgeJ,
            degreeNormalizeI = true,
            degreeNormalizeJ = true,
            aggregateJ = true
          )
          .value
          .toMat
          .roundTo(4) == Mat(
          Vec(math.sqrt(2d), math.sqrt(2d) / 2d, math.sqrt(2d) / 2d),
          Vec(math.sqrt(2d), math.sqrt(2d) / 2d, math.sqrt(2d) / 2d),
          Vec(math.sqrt(2d), math.sqrt(2d) / 2d, math.sqrt(2d) / 2d)
        ).roundTo(4)
      )

    }
  }

}
