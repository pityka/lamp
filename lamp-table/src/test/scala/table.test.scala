package lamp.table

import lamp._

import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.compatible.Assertion
import org.saddle._
import java.nio.channels.Channels
import java.io.ByteArrayInputStream
import org.saddle.index._
import lamp.saddle._

class TableSuite extends AnyFunSuite {
  implicit def AssertionIsMovable: EmptyMovable[Assertion] = Movable.empty[Assertion]

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

  val csvText5 = """hint,hfloat,htime,hbool,htext
1,1.5,2020-01-01T00:00:00Z,false,"something, something"
NA,NA,NA,NA,NA
2,3.0,2021-01-01T00:00:00Z,true,"a,""""

  def makeTable5(implicit scope: Scope) = {
    val channel =
      Channels.newChannel(new ByteArrayInputStream(csvText5.getBytes()))

    lamp.table.csv
      .readHeterogeneousFromCSVChannel(
        List(
          (0, I64ColumnType),
          (1, F32ColumnType),
          (2, DateTimeColumnType()),
          (3, BooleanColumnType()),
          (4, TextColumnType(64, -1L, None))
        ),
        channel = channel,
        recordSeparator = "\n",
        header = true
      )
      .toOption
      .get
  }
  def makeTable3(implicit scope: Scope) = {
    val channel =
      Channels.newChannel(new ByteArrayInputStream(csvText3.getBytes()))

    lamp.table.csv
      .readHeterogeneousFromCSVChannel(
        List(
          (0, I64ColumnType),
          (1, F32ColumnType)
        ),
        channel = channel,
        recordSeparator = "\n",
        header = true
      )
      .toOption
      .get
  }
  def makeTable4(implicit scope: Scope) = {
    val channel =
      Channels.newChannel(new ByteArrayInputStream(csvText4.getBytes()))

    lamp.table.csv
      .readHeterogeneousFromCSVChannel(
        List(
          (0, I64ColumnType),
          (1, TextColumnType(2, -1, None)),
          (2, TextColumnType(2, -1, None))
        ),
        channel = channel,
        recordSeparator = "\n",
        header = true
      )
      .toOption
      .get
  }
  def makeTable2(implicit scope: Scope) = {
    val channel =
      Channels.newChannel(new ByteArrayInputStream(csvText2.getBytes()))

    lamp.table.csv
      .readHeterogeneousFromCSVChannel(
        List(
          (0, I64ColumnType),
          (1, F32ColumnType)
        ),
        channel = channel,
        recordSeparator = "\n",
        header = true
      )
      .toOption
      .get
  }
  def makeTable1(implicit scope: Scope) = {
    val channel =
      Channels.newChannel(new ByteArrayInputStream(csvText.getBytes()))
    lamp.table.csv
      .readHeterogeneousFromCSVChannel(
        List(
          (0, I64ColumnType),
          (1, F32ColumnType),
          (2, DateTimeColumnType()),
          (3, BooleanColumnType()),
          (4, TextColumnType(64, -1L, None))
        ),
        channel = channel,
        recordSeparator = "\n",
        header = true
      )
      .toOption
      .get
  }

  test("pivot") {
    Scope.root { implicit scope =>
      val table = makeTable4
      val pivoted = table.pivot(0, 1)(_.colsAt(2).rows(0))
      assert(pivoted.numRows == 4)
      assert(pivoted.numCols == 5)
      assert(pivoted.firstCol("a").toVec == Vec("v1", null, "v5", null))
      ()
    }
  }

  test("outer join") {
    Scope.root { implicit scope =>
      val table = makeTable3
      val table2 = makeTable2


      val right = table.equijoin(0, table2, 0, RightJoin)
      val left = table.equijoin(0, table2, 0, LeftJoin)
      val outer = table.equijoin(0, table2, 0, OuterJoin)
      assert(right.numRows == 3)
      assert(left.numRows == 3)
      assert(outer.numRows == 4)
      assert(right.numCols == 3)
      assert(left.numCols == 3)
      assert(outer.numCols == 3)
      assert(
        right.firstCol("hfloat2").values.toVec.toString == Vec(
          0.5,
          Double.NaN,
          0.5
        ).toString
      )
      assert(
        right.firstCol("hfloat").values.toVec.toString == Vec(
          5.5,
          4.5,
          6.0
        ).toString
      )
      assert(
        left.firstCol("hfloat2").values.toVec.toString == Vec(
          0.5,
          0.5,
          1.5
        ).toString
      )
      assert(
        left.firstCol("hfloat").values.toVec.toString == Vec(
          5.5,
          6.0,
          Double.NaN
        ).toString
      )
      assert(
        outer.firstCol("hfloat2").values.toVec.toString == Vec(
          0.5,
          0.5,
          1.5,
          Double.NaN
        ).toString
      )
      assert(
        outer.firstCol("hfloat").values.toVec.toString == Vec(
          5.5,
          6.0,
          Double.NaN,
          4.5
        ).toString
      )
      ()
    }
  }

  test("row with missing") {
    Scope.root { implicit scope =>
      val table = makeTable1
      val selected = table.rows(Array(-1, 1, -1, 2, 1))
      assert(
        selected.colAt(1).values.toFloatVec.toString == Vec(
          Float.NaN,
          2.5f,
          Float.NaN,
          3.0f,
          2.5f
        ).toString
      )
      ()
    }
  }

  test("theta join") {
    Scope.root { implicit scope =>
      val table = makeTable1
      val table2 = makeTable2
      
      val joined = table.join(table2,2){t => Column((t(1).values+3d) gt t(6).values)}
      assert(joined.numRows == 3)
      assert(joined.numCols == 7)
      assert(joined.colAt(1).values.toFloatVec == Vec(2.5,3.0,3.0))
      assert(joined.colAt(6).values.toFloatVec == Vec(4.5,5.5,4.5))
      ()
    }
  }
  test("join") {
    Scope.root { implicit scope =>
      val table = makeTable1
      val table2 = makeTable2
      val joined = table.equijoin("hint", table2, "hint", InnerJoin)
      assert(joined.numRows == 5)
      assert(joined.numCols == 6)
      assert(joined.colAt(5).values.toFloatVec == Vec(4.5, 5.5, 6.0, 5.5, 6.0))
      assert(joined.colAt(0).values.toLongVec == Vec(1L, 2L, 2L, 2L, 2L))
      ()
    }
  }
  test("product") {
    Scope.root { implicit scope =>
      val table = makeTable1
      val table2 = makeTable2
      val product = table.product(table2)

      assert(product.numRows == 9)
      assert(product.numCols == 7)
      assert(
        product.colAt(6).values.toFloatVec == Vec(5.5, 4.5, 6.0, 5.5, 4.5, 6.0,
          5.5, 4.5, 6.0)
      )
      assert(
        product.colAt(1).values.toFloatVec == Vec(1.5, 1.5, 1.5, 2.5, 2.5, 2.5,
          3d, 3d, 3d)
      )
      ()
    }
  }
  test("distinct") {
    Scope.root { implicit scope =>
      val table = makeTable1
      val table2 = makeTable2
      val distinct = table.product(table2).colsAt(0, 1).distinct

      assert(distinct.numRows == 3)
      assert(distinct.numCols == 2)
      assert(distinct.colAt(0).values.toLongVec == Vec(1L, 2L, 2L))
      assert(distinct.colAt(1).values.toFloatVec == Vec(1.5, 2.5, 3.0))
      ()
    }
  }
  test("filter") {
    Scope.root { implicit scope =>
      val table = makeTable1

      val filtered = table.filter(table.colAt(0).equ(2L))

      assert(filtered.numRows == 2)
      assert(filtered.numCols == 5)
      assert(filtered.colAt(0).values.toLongVec == Vec(2L, 2L))
      assert(filtered.colAt(1).values.toFloatVec == Vec(2.5, 3.0))
      ()
    }
  }
  test("equi filter") {
    Scope.root { implicit scope =>
      val table = makeTable1
      val filtered = table.equifilter(t => t("htext") === "something,")

      assert(filtered.numRows == 1)
      assert(filtered.numCols == 5)
      assert(filtered.colAt(0).values.toLongVec == Vec(2L))
      assert(filtered.colAt(1).values.toFloatVec == Vec(2.5))
      ()
    }
  }
  test("rfilter") {
    Scope.root { implicit scope =>
      val table = makeTable1
      val filtered = table.rfilter()(r => r.first("htext").get.asInstanceOf[String] == "something,")

      assert(filtered.numRows == 1)
      assert(filtered.numCols == 5)
      assert(filtered.colAt(0).values.toLongVec == Vec(2L))
      assert(filtered.colAt(1).values.toFloatVec == Vec(2.5))
      ()
    }
  }

  test("aggregate") {
    Scope.root { implicit scope =>
      val table = makeTable1

      val aggregated =
        table
          .union(table)
          .groupBy(
            "hint",
            table.colNames.getFirst("hfloat")
          )
          .project("hfloat")
          .aggregate { group =>
            Column(group.colAt(0).values.sum(dim = 0, keepDim = true)).table

          }

      assert(aggregated.numRows == 3)
      assert(aggregated.numCols == 1)
      assert(aggregated.colAt(0).values.toFloatVec == Vec(3d, 5d, 6d))
      ()
    }
  }
  test("union") {
    Scope.root { implicit scope =>
      val table = makeTable1

      val unioned = table.union(table, table)
      assert(unioned.numRows == table.numRows * 3)
      assert(unioned.numCols == table.numCols)
      assert(
        unioned.colAt(1).values.toFloatVec == Vec(1.5f, 2.5f, 3.0f, 1.5f, 2.5f,
          3.0f, 1.5f, 2.5, 3.0f)
      )
      ()
    }
  }

  test("group by") {
    Scope.root { implicit scope =>
      val table = makeTable1

      val grouped =
        table
          .groupBy(0)
          .aggregate(group =>
            Table(Column(group.colAt(1).values.mean.view(-1)))
          )
      assert(grouped.numRows == 2)
      assert(grouped.numCols == 1)
      assert(grouped.colAt(0).values.toFloatVec == Vec(1.5, 2.75))
      ()
    }
  }
  test("without col") {
    Scope.root { implicit scope =>
      val table = makeTable1
        .withoutCol(Set(0))

      assert(table.numRows == 3L)
      assert(table.numCols == 4)
      assert(
        table.colNames == Index(
          ("hfloat"),
          ("htime"),
          ("hbool"),
          ("htext")
        )
      )
      assert(table.colAt(0).values.toFloatVec == Vec(1.5f, 2.5f, 3.0f))
      assert(
        table.colAt(1).values.toLongVec == Vec(
          java.time.Instant.parse("2020-01-01T00:00:00Z").toEpochMilli(),
          java.time.Instant.parse("2021-01-01T00:00:00Z").toEpochMilli(),
          java.time.Instant.parse("2021-01-01T00:00:00Z").toEpochMilli()
        )
      )
      assert(table.colAt(2).values.toLongVec == Vec(0L, 1L, 1L))
      assert(
        table
          .colAt(3)
          .values
          .toLongMat
          .rows
          .map(v => v.filter(_ >= 0).map(_.toChar).toSeq.mkString) == Seq(
          "something, something",
          "something,",
          "a,"
        )
      )
      ()
    }
  }
  test("csv reader") {
    Scope.root { implicit scope =>
      val table = makeTable1

      assert(table.numRows == 3L)
      assert(table.numCols == 5)
      assert(
        table.colNames == Index(
          ("hint"),
          ("hfloat"),
          ("htime"),
          ("hbool"),
          ("htext")
        )
      )
      assert(table.colAt(0).values.toLongVec == Vec(1L, 2L, 2L))
      assert(table.colAt(1).values.toFloatVec == Vec(1.5f, 2.5f, 3.0f))
      assert(
        table.colAt(2).values.toLongVec == Vec(
          java.time.Instant.parse("2020-01-01T00:00:00Z").toEpochMilli(),
          java.time.Instant.parse("2021-01-01T00:00:00Z").toEpochMilli(),
          java.time.Instant.parse("2021-01-01T00:00:00Z").toEpochMilli()
        )
      )
      assert(table.colAt(3).values.toLongVec == Vec(0L, 1L, 1L))
      assert(
        table
          .colAt(4)
          .values
          .toLongMat
          .rows
          .map(v => v.filter(_ >= 0).map(_.toChar).toSeq.mkString) == Seq(
          "something, something",
          "something,",
          "a,"
        )
      )
      ()
    }
  }
  test("csv writer") {
    Scope.root { implicit scope =>
      val table = makeTable1
      val rendered = lamp.table.csv.renderToCSVString(table, recordSeparator = "\n").dropRight(1)
      assert(csvText == rendered)
      ()
    }
  }
  test("equality") {
    Scope.root { implicit scope =>
      val table = makeTable1

      val table2 = makeTable1

      assert(table == table)
      assert(table != table2)
      assert(table equalDeep table2)
      assert(table equalDeep table)
      ()
    }
  }
  test("missingness") {
    Scope.root { implicit scope =>
      val table = makeTable5

      assert(
        table.colAt(0).missingnessMask.castToLong.toVec == Vec(0L, 1L, 0L)
      )
      assert(
        table.colAt(1).missingnessMask.castToLong.toVec == Vec(0L, 1L, 0L)
      )
      assert(
        table.colAt(2).missingnessMask.castToLong.toVec == Vec(0L, 1L, 0L)
      )
      assert(
        table.colAt(3).missingnessMask.castToLong.toVec == Vec(0L, 1L, 0L)
      )
      assert(
        table.colAt(4).missingnessMask.castToLong.toVec == Vec(0L, 1L, 0L)
      )
      ()
    }
  }
  test("binary writer") {
    Scope.root { implicit scope =>
      val table = makeTable1
      val tmp = java.io.File.createTempFile("test","table")
      import cats.effect.unsafe.implicits.global

      lamp.table.io.writeTableToFile(table,tmp).unsafeRunSync().toOption.get 
      val readBack = lamp.table.io.readTableFromFile(tmp)
      assert(readBack.equalDeep(table))
      tmp.delete
      ()
    }
  }
}
