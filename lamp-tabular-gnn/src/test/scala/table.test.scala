package lamp.tgnn

import lamp._

import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.compatible.Assertion
import org.saddle._
import java.nio.channels.Channels
import java.io.ByteArrayInputStream
import org.saddle.index._

class TableSuite extends AnyFunSuite {
  import Table._
  implicit def AssertionIsMovable = Movable.empty[Assertion]

  val csvText = """hint,hfloat,htime,hbool,htext
1,1.5,2020-01-01T00:00:00Z,false,"something, something"
2,2.5,2021-01-01T00:00:00Z,true,"something,"
2,3.0,2021-01-01T00:00:00Z,true,"a,""""
  val csvText2 = """hint,hfloat
2,5.5
1,4.5
2,6.0"""
  val csvText3 = """hint,hfloat2
2,0.5
3,1.5"""
  val csvText4 = """key,pivot,value
0,"a","v1"
0,"b","v2"
0,"b","v3"
1,"c","v4"
2,"a","v5"
0,"c","v6"
3,"d","v7"
"""

test("pivot") {
    Scope.root { implicit scope =>
      val channel =
        Channels.newChannel(new ByteArrayInputStream(csvText4.getBytes()))
     
      val table = Table
        .readHeterogeneousFromCSVChannel(
          List(
            (0, I64Column),
            (1, TextColumn(2,-1,None)),
            (2, TextColumn(2,-1,None))
          ),
          channel = channel,
          recordSeparator = "\n",
          header = true
        )
        .toOption
        .get
        val pivoted = table.pivot[Long,String](0,1)(_.cols(2).rows(0))
        println(pivoted.stringify())
    }
  }

 test("outer join") {
    Scope.root { implicit scope =>
      val channel =
        Channels.newChannel(new ByteArrayInputStream(csvText3.getBytes()))
      val channel2 =
        Channels.newChannel(new ByteArrayInputStream(csvText2.getBytes()))
      val table = Table
        .readHeterogeneousFromCSVChannel(
          List(
            (0, I64Column),
            (1, F32Column)
          ),
          channel = channel,
          recordSeparator = "\n",
          header = true
        )
        .toOption
        .get
      val table2 = Table
        .readHeterogeneousFromCSVChannel(
          List(
            (0, I64Column),
            (1, F32Column)
          ),
          channel = channel2,
          recordSeparator = "\n",
          header = true
        )
        .toOption
        .get
      val right = table.join(0, table2, 0, RightJoin)
      val left = table.join(0, table2, 0, LeftJoin)
      val outer = table.join(0, table2, 0, OuterJoin)
      assert(right.numRows == 3)
      assert(left.numRows == 3)
      assert(outer.numRows == 4)
      assert(right.numCols == 3)
      assert(left.numCols == 3)
      assert(outer.numCols == 3)
      assert(right.col("hfloat2").toVec.toString == Vec(0.5,Double.NaN,0.5).toString)
      assert(right.col("hfloat").toVec.toString == Vec(5.5,4.5,6.0).toString)
      assert(left.col("hfloat2").toVec.toString == Vec(0.5,0.5,1.5).toString)
      assert(left.col("hfloat").toVec.toString == Vec(5.5,6.0,Double.NaN).toString)
      assert(outer.col("hfloat2").toVec.toString == Vec(0.5,0.5,1.5,Double.NaN).toString)
      assert(outer.col("hfloat").toVec.toString == Vec(5.5,6.0,Double.NaN,4.5).toString)
      
    }
  }

  test("row with missing") {
    Scope.root { implicit scope =>
      val channel =
        Channels.newChannel(new ByteArrayInputStream(csvText.getBytes()))
      val table = Table
        .readHeterogeneousFromCSVChannel(
          List(
            (0, I64Column),
            (1, F32Column),
            (2, DateTimeColumn()),
            (3, BooleanColumn()),
            (4, TextColumn(64, -1L, None))
          ),
          channel = channel,
          recordSeparator = "\n",
          header = true
        )
        .toOption
        .get

      val selected = table.rows(Array(-1, 1, -1, 2, 1))
      assert(
        selected.col(1).toFloatVec.toString == Vec(
          Float.NaN,
          2.5f,
          Float.NaN,
          3.0f,
          2.5f
        ).toString
      )

    }
  }

  test("join") {
    Scope.root { implicit scope =>
      val channel =
        Channels.newChannel(new ByteArrayInputStream(csvText.getBytes()))
      val channel2 =
        Channels.newChannel(new ByteArrayInputStream(csvText2.getBytes()))
      val table = Table
        .readHeterogeneousFromCSVChannel(
          List(
            (0, I64Column),
            (1, F32Column),
            (2, DateTimeColumn()),
            (3, BooleanColumn()),
            (4, TextColumn(64, -1L, None))
          ),
          channel = channel,
          recordSeparator = "\n",
          header = true
        )
        .toOption
        .get
      val table2 = Table
        .readHeterogeneousFromCSVChannel(
          List(
            (0, I64Column),
            (1, F32Column)
          ),
          channel = channel2,
          recordSeparator = "\n",
          header = true
        )
        .toOption
        .get
      val joined = table.join(0, table2, 0)
      assert(joined.numRows == 5)
      assert(joined.numCols == 6)
      assert(joined.col(5).toFloatVec == Vec(4.5, 5.5, 6.0, 5.5, 6.0))
      assert(joined.col(0).toLongVec == Vec(1L, 2L, 2L, 2L, 2L))
    }
  }
  test("union") {
    Scope.root { implicit scope =>
      val channel =
        Channels.newChannel(new ByteArrayInputStream(csvText.getBytes()))
      val table = Table
        .readHeterogeneousFromCSVChannel(
          List(
            (0, I64Column),
            (1, F32Column),
            (2, DateTimeColumn()),
            (3, BooleanColumn()),
            (4, TextColumn(64, -1L, None))
          ),
          channel = channel,
          recordSeparator = "\n",
          header = true
        )
        .toOption
        .get

      val unioned = table.union(table, table)
      assert(unioned.numRows == table.numRows * 3)
      assert(unioned.numCols == table.numCols)
      assert(
        unioned.col(1).toFloatVec == Vec(1.5f, 2.5f, 3.0f, 1.5f, 2.5f, 3.0f,
          1.5f, 2.5, 3.0f)
      )

    }
  }

  test("group by") {
    Scope.root { implicit scope =>
      val channel =
        Channels.newChannel(new ByteArrayInputStream(csvText.getBytes()))

      val table = Table
        .readHeterogeneousFromCSVChannel(
          List(
            (0, I64Column),
            (1, F32Column),
            (2, DateTimeColumn()),
            (3, BooleanColumn()),
            (4, TextColumn(64, -1L, None))
          ),
          channel = channel,
          recordSeparator = "\n",
          header = true
        )
        .toOption
        .get

      val grouped =
        table.groupByThenUnion[Long](0)((_,table) => Table.unnamed(table.col(1).mean.view(-1)))
      assert(grouped.numRows == 2)
      assert(grouped.numCols == 1)
      assert(grouped.col(0).toFloatVec == Vec(1.5, 2.75))

    }
  }
  test("without col") {
    Scope.root { implicit scope =>
      val channel =
        Channels.newChannel(new ByteArrayInputStream(csvText.getBytes()))

      val table = Table
        .readHeterogeneousFromCSVChannel(
          List(
            (0, I64Column),
            (1, F32Column),
            (2, DateTimeColumn()),
            (3, BooleanColumn()),
            (4, TextColumn(64, -1L, None))
          ),
          channel = channel,
          recordSeparator = "\n",
          header = true
        )
        .toOption
        .get
        .withoutCol(Set(0))

      assert(table.numRows == 3L)
      assert(table.numCols == 4)
      assert(
        table.colNames == Vector(
          Some("hfloat"),
          Some("htime"),
          Some("hbool"),
          Some("htext")
        )
      )
      assert(table.col(0).toFloatVec == Vec(1.5f, 2.5f, 3.0f))
      assert(
        table.col(1).toLongVec == Vec(
          java.time.Instant.parse("2020-01-01T00:00:00Z").toEpochMilli(),
          java.time.Instant.parse("2021-01-01T00:00:00Z").toEpochMilli(),
          java.time.Instant.parse("2021-01-01T00:00:00Z").toEpochMilli()
        )
      )
      assert(table.col(2).toLongVec == Vec(0L, 1L, 1L))
      assert(
        table
          .col(3)
          .toLongMat
          .rows
          .map(v => v.filter(_ >= 0).map(_.toChar).toSeq.mkString) == Seq(
          "something, something",
          "something,",
          "a,"
        )
      )

    }
  }
  test("csv reader") {
    Scope.root { implicit scope =>
      val channel =
        Channels.newChannel(new ByteArrayInputStream(csvText.getBytes()))

      val table = Table
        .readHeterogeneousFromCSVChannel(
          List(
            (0, I64Column),
            (1, F32Column),
            (2, DateTimeColumn()),
            (3, BooleanColumn()),
            (4, TextColumn(64, -1L, None))
          ),
          channel = channel,
          recordSeparator = "\n",
          header = true
        )
        .toOption
        .get

      assert(table.numRows == 3L)
      assert(table.numCols == 5)
      assert(
        table.colNames == Vector(
          Some("hint"),
          Some("hfloat"),
          Some("htime"),
          Some("hbool"),
          Some("htext")
        )
      )
      assert(table.col(0).toLongVec == Vec(1L, 2L, 2L))
      assert(table.col(1).toFloatVec == Vec(1.5f, 2.5f, 3.0f))
      assert(
        table.col(2).toLongVec == Vec(
          java.time.Instant.parse("2020-01-01T00:00:00Z").toEpochMilli(),
          java.time.Instant.parse("2021-01-01T00:00:00Z").toEpochMilli(),
          java.time.Instant.parse("2021-01-01T00:00:00Z").toEpochMilli()
        )
      )
      assert(table.col(3).toLongVec == Vec(0L, 1L, 1L))
      assert(
        table
          .col(4)
          .toLongMat
          .rows
          .map(v => v.filter(_ >= 0).map(_.toChar).toSeq.mkString) == Seq(
          "something, something",
          "something,",
          "a,"
        )
      )

    }
  }
}
