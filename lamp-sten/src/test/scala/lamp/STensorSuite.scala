package lamp

import org.saddle._
import org.saddle.linalg._
import org.saddle.ops.BinOps._

import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.compatible.Assertion

class STenSuite extends AnyFunSuite {
  implicit def AssertionIsMovable = Movable.empty[Assertion]

  test("cast to half") {
    Scope.root { implicit scope =>
      val half = STen.ones(List(3)).castToHalf
      assert(half.scalarTypeByte == 5)

    }
  }

  test("unique 1") {
    Scope.root { implicit scope =>
      val t =
        STen.fromLongMat(Mat(Vec(3, 3, 9), Vec(8, 8, 1)).map(_.toLong), false)
      val (un, _) = t.unique(sorted = false, returnInverse = false)
      assert(un.toLongVec.sorted == Vec(1, 3, 8, 9))
    }
  }
  test("unique 3") {
    Scope.root { implicit scope =>
      val t =
        STen.fromLongVec(Vec(1, 1, 2, 3, 4, 4).map(_.toLong), false)
      val (un, _) = t.unique(sorted = true, returnInverse = false)
      assert(un.toLongVec == Vec(1, 2, 3, 4))
    }
  }
  test("unique 4") {
    Scope.root { implicit scope =>
      val t =
        STen.fromLongVec(Vec(1, 1, 2, 3, 4, 4).map(_.toLong), false)
      val (un, _, _) = t.uniqueConsecutive(dim = 0, returnInverse = false)
      assert(un.toLongVec == Vec(1, 2, 3, 4))
    }
  }
  test("unique 2") {
    Scope.root { implicit scope =>
      val t =
        STen.fromLongMat(Mat(Vec(3, 3, 9), Vec(8, 8, 1)).map(_.toLong), false)
      val (un, inv, count) = t.unique(
        sorted = false,
        returnInverse = true,
        dim = 0,
        returnCounts = true
      )
      assert(un.toLongMat == Mat(Vec(3, 9), Vec(8, 1)))
      assert(inv.toLongVec == Vec(0, 0, 1))
      assert(count.toLongVec == Vec(2, 1))
    }
  }

  test("zeros cpu") {
    Scope.root { implicit scope =>
      val sum = Scope { implicit scope =>
        val ident = STen.eye(3, STenOptions.d)
        val ones = STen.ones(List(3, 3), STenOptions.d)
        ident + ones
      }
      assert(sum.toMat == mat.ones(3, 3) + mat.ident(3))
    }
  }

  test("t") {
    Scope.root { implicit scope =>
      val t1 = STen.ones(List(3, 2), STenOptions.d)
      assert(t1.t.shape == List(2, 3))
    }
  }
  test("transpose") {
    Scope.root { implicit scope =>
      val t1 = STen.ones(List(3, 2, 4), STenOptions.d)
      assert(t1.transpose(1, 2).shape == List(3, 4, 2))
    }
  }
  test("select") {
    Scope.root { implicit scope =>
      val t1 = STen.ones(List(3, 2, 4), STenOptions.d)
      assert(t1.select(1, 1).shape == List(3, 4))
    }
  }
  test("indexSelect") {
    Scope.root { implicit scope =>
      val t1 = STen.ones(List(3, 2, 4), STenOptions.d)
      assert(
        t1.indexSelect(0, STen.fromLongVec(Vec(0L, 1L), false)).shape == List(
          2,
          2,
          4
        )
      )
    }
  }
  test("argmax") {
    Scope.root { implicit scope =>
      val t1 = STen.eye(3, STenOptions.d)
      assert(
        t1.argmax(0, keepDim = false).toLongVec == Vec(0, 1, 2)
      )
    }
  }
  test("argmin") {
    Scope.root { implicit scope =>
      val t1 = STen.eye(3, STenOptions.d).neg
      assert(
        t1.argmin(0, keepDim = false).toLongVec == Vec(0, 1, 2)
      )
    }
  }
  test("maskFill") {
    Scope.root { implicit scope =>
      val t1 = STen.eye(3, STenOptions.d)
      val mask = t1.equ(0d)
      assert(
        t1.maskFill(mask, -1).toMat.row(0) == Vec(1d, -1d, -1d)
      )
    }
  }
  test("maskScatter") {
    Scope.root { implicit scope =>
      val t1 = STen.eye(3, STenOptions.d)
      val t2 = STen.zeros(List(3, 3), STenOptions.d)
      val mask = t1.equ(1d)
      assert(
        t2.maskedScatter(mask, t1 * 4).toMat.row(0) == Vec(4d, 0d, 0d)
      )
    }
  }
  test("cat") {
    Scope.root { implicit scope =>
      val t1 = STen.eye(3, STenOptions.d)
      assert(
        t1.cat(t1, 0).shape == List(6, 3)
      )
    }
  }
  test("cast to long") {
    Scope.root { implicit scope =>
      val t1 = STen.eye(2, STenOptions.d)
      assert(
        t1.castToLong.toLongMat == mat.ident(2).map(_.toLong)
      )
    }
  }
  test("cast to double") {
    Scope.root { implicit scope =>
      val t1 = STen.eye(2, STenOptions.l)
      assert(
        t1.castToDouble.toMat == mat.ident(2)
      )
    }
  }
  test("+") {
    Scope.root { implicit scope =>
      val t1 = STen.eye(2, STenOptions.d)
      assert(
        (t1 + t1 + t1) equalDeep (t1 * 3)
      )
    }
  }
  test("add") {
    Scope.root { implicit scope =>
      val t1 = STen.eye(2, STenOptions.d)
      assert(
        (t1.add(t1, 2d)) equalDeep (t1 * 3)
      )
    }
  }
  test("add2") {
    Scope.root { implicit scope =>
      val t1 = STen.zeros(List(2, 2), STenOptions.d)
      assert(
        (t1.add(1d, 1d)) equalDeep (STen.ones(List(2, 2), STenOptions.d))
      )
    }
  }
  test("+-") {
    Scope.root { implicit scope =>
      val t1 = STen.eye(2, STenOptions.d)
      assert(
        (t1 + t1 - t1) equalDeep (t1)
      )
    }
  }
  test("sub") {
    Scope.root { implicit scope =>
      val t1 = STen.ones(List(2, 2), STenOptions.d)
      assert(
        ((t1 + t1) sub (t1, 2d)) equalDeep (STen
          .zeros(List(2, 2), STenOptions.d))
      )
    }
  }
  test("*") {
    Scope.root { implicit scope =>
      val t1 = STen.ones(List(2, 2), STenOptions.d)
      val t0 = STen.zeros(List(2, 2), STenOptions.d)
      assert(
        (t1 * t0) equalDeep t0
      )
    }
  }
  test("/") {
    Scope.root { implicit scope =>
      val t1 = STen.ones(List(2, 2), STenOptions.d)
      val t2 = STen.ones(List(2, 2), STenOptions.d) * 2
      assert(
        (t1 / t2).sum.toMat.raw(0, 0) == 2
      )
    }
  }
  test("/2") {
    Scope.root { implicit scope =>
      val t1 = STen.ones(List(2, 2), STenOptions.d)
      assert(
        (t1 / 2).sum.toMat.raw(0, 0) == 2
      )
    }
  }
  test("mm") {
    Scope.root { implicit scope =>
      val t1 = STen.rand(List(2, 2), STenOptions.d)
      val t2 = STen.rand(List(2, 2), STenOptions.d)
      assert(
        (t1 mm t2).toMat.roundTo(4) == (t1.toMat mm t2.toMat).roundTo(4)
      )
    }
  }
  test("matmul 2") {
    Scope.root { implicit scope =>
      val t1 = STen.rand(List(2, 2), STenOptions.d)
      val t2 = STen.rand(List(2, 2), STenOptions.d)
      assert(
        (t1 matmul t2).toMat.roundTo(4) == (t1.toMat mm t2.toMat).roundTo(4)
      )
    }
  }
  test("matmul 3") {
    Scope.root { implicit scope =>
      val t1 = STen.rand(List(100, 10, 3), STenOptions.d)
      val t2 = STen.rand(List(3), STenOptions.d)
      assert(
        (t1 matmul t2).shape == List(100, 10)
      )
    }
  }
  test("matmul") {
    Scope.root { implicit scope =>
      val t1 = STen.rand(List(2, 2), STenOptions.d)
      val t2 = STen.rand(List(2), STenOptions.d)
      assert(
        (t1 matmul t2).toVec.roundTo(4) == (t1.toMat mv t2.toVec).roundTo(4)
      )
    }
  }
  test("dot") {
    Scope.root { implicit scope =>
      val t1 = STen.rand(List(4), STenOptions.d)
      val t2 = STen.rand(List(4), STenOptions.d)
      assert(
        (t1 dot t2).toVec.roundTo(4) == Vec(
          t1.toVec dot t2.toVec
        ).roundTo(4)
      )
    }
  }
  test("bmm") {
    Scope.root { implicit scope =>
      val t1 = STen.rand(List(1, 2, 2), STenOptions.d)
      val t2 = STen.rand(List(1, 2, 2), STenOptions.d)
      assert(
        (t1 bmm t2)
          .view(2, 2)
          .toMat == (t1.view(2, 2).toMat mm t2.view(2, 2).toMat)
      )
    }
  }
  test("relu") {
    Scope.root { implicit scope =>
      val t1 = STen.eye(3, STenOptions.d).neg
      assert(
        t1.relu equalDeep (STen.zeros(List(3, 3)))
      )
    }
  }
  test("exp log") {
    Scope.root { implicit scope =>
      val t1 = STen.ones(List(3, 3))
      assert(
        t1.exp.log.round equalDeep t1
      )
    }
  }
  test("square sqrt") {
    Scope.root { implicit scope =>
      val t1 = STen.ones(List(3, 3)) + 1
      assert(
        t1.square.sqrt.round equalDeep t1
      )
    }
  }
  test("abs") {
    Scope.root { implicit scope =>
      val t1 = STen.ones(List(3, 3)).neg
      assert(
        t1.abs equalDeep t1.neg
      )
    }
  }
  test("floor") {
    Scope.root { implicit scope =>
      val t1 = STen.ones(List(3, 3)) + 0.7
      assert(
        t1.floor equalDeep STen.ones(List(3, 3))
      )
    }
  }
  test("ceil") {
    Scope.root { implicit scope =>
      val t1 = STen.ones(List(3, 3)) + 0.7
      assert(
        t1.ceil equalDeep (STen.ones(List(3, 3)) + 1)
      )
    }
  }
  test("reciprocal") {
    Scope.root { implicit scope =>
      val t1 = STen.rand(List(3, 3))
      assert(
        t1.reciprocal equalDeep t1.pow(-1)
      )
    }
  }
  test("trace") {
    Scope.root { implicit scope =>
      val t1 = STen.eye(3)
      assert(
        t1.trace.toMat.raw(0) == 3d
      )
    }
  }
  test("npy") {
    Scope.root { implicit scope =>
      val read =
        lamp.io.npy.readDoubleFromChannel(
          java.nio.channels.Channels
            .newChannel(getClass.getResourceAsStream("/file.npy")),
          CPU
        )
      assert(read.shape == List(3, 3))
      assert(
        read.toMat == (Mat(Vec(1d, 0d, 0d), Vec(0d, 1d, 0d), Vec(0d, 0d, 1d)))
      )
    }

  }

  test("csv") {
    Scope.root { implicit scope =>
      val tmp = java.io.File.createTempFile("dfsd", ".csv")
      val is = new java.io.FileInputStream(tmp)
      val writer = new java.io.BufferedWriter(new java.io.FileWriter(tmp))
      writer.write("a,b\r\n")
      val line = "1.0,2.0\r\n"
      val numRows = 1000L
      var i = 0L
      while (i < numRows) {
        writer.write(line)
        i += 1
      }
      writer.close
      val channel = is.getChannel
      val (h, ten) =
        lamp.io.csv.readFromChannel(6, channel, CPU, header = true).toOption.get
      assert(h == Some(List("a", "b")))
      assert(ten.shape == List(numRows, 2))
    }
  }

  test("csv - buffer") {
    val buffer =
      new lamp.io.csv.Buffer[Double](Array.ofDim[Double](0), 0, Nil, 5)
    0 until 100 foreach (i => buffer.+=(i.toDouble))
    assert(
      buffer.toArrays.reduce(_ ++ _).toList == (0 until 100).toList
        .map(_.toDouble)
    )
  }

}
