package lamp.tgnn

import org.scalatest.funsuite.AnyFunSuite
import java.nio.channels.Channels
import Table._
import lamp._
import java.io.ByteArrayInputStream
import org.saddle._
class RelationAlgebraSuite extends AnyFunSuite {
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

  test("dfs") {
    Scope.root { implicit scope =>
      val table = Table
        .readHeterogeneousFromCSVChannel(
          List(
            (0, I64Column),
            (1, F32Column),
            (2, DateTimeColumn()),
            (3, BooleanColumn()),
            (4, TextColumn(64, -1L, None))
          ),
          channel =
            Channels.newChannel(new ByteArrayInputStream(csvText.getBytes())),
          recordSeparator = "\n",
          header = true
        )
        .toOption
        .get

      val table3 = Table
        .readHeterogeneousFromCSVChannel(
          List(
            (0, I64Column),
            (1, F32Column)
          ),
          channel =
            Channels.newChannel(new ByteArrayInputStream(csvText3.getBytes())),
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
          channel =
            Channels.newChannel(new ByteArrayInputStream(csvText2.getBytes())),
          recordSeparator = "\n",
          header = true
        )
        .toOption
        .get

      println(table.stringify())
      println(table2.stringify())
      println(table3.stringify())

      val tref1 = Q.table("t1")
      val tref2 = Q.table("t2")
      val noop = tref1.scan.done.bind(tref1 -> table)
      assert(noop.interpret == table)
      assert(noop.interpret != table2)
      assert(noop.bind(tref1 -> table2).interpret == table2)

      val project = tref1.scan
        .project(tref1.col(0).self, tref1.col("hbool").self)
        .project(tref1.col("hbool").self)
        .done
        .bind(tref1 -> table)
      assert(project.interpret == table.cols(3))

      val predicate = tref1.col("hbool") === 0

      val filter = tref1.scan
        .filter(predicate.negate.or(predicate))
        .done
        .bind(tref1 -> table)
      assert(filter.interpret equalDeep table)
      assert(filter.interpret != table)

      val innerJoin =
        tref1.scan
          .innerEquiJoin(
            tref1.col("hint"),
            tref2.scan,
            tref2.col("hint")
          )
          .done
          .bind(tref1 -> table, tref2 -> table2)

      assert(
        innerJoin.interpret equalDeep table
          .join(0, table2, 0)
      )

      assert(
        Q.query(table) { tref1 =>
          Q.query(table2) { tref2 =>
            tref1.scan
              .innerEquiJoin(
                tref1.col("hint"),
                tref2.scan,
                tref2.col("hint")
              )
              .filter(tref2.col("hfloat") === 4.5)
              .done
          }
        }.interpret
          .col(5)
          .values
          .toMat == Mat(Vec(4.5))
      )
      assert(
        Q.query(table, "t1") { tref1 =>
          Q.query(table2, "t2") { tref2 =>
            tref1.scan
              .product(
                tref2.asOp
              )
              .done
          }
        }.interpret
          .col(6)
          .toVec == Vec(5.5, 4.5, 6.0, 5.5, 4.5, 6.0, 5.5, 4.5, 6.0)
        // .toMat == Mat(Vec(4.5))
      )

      assert(
        Q.query(table, "t1") { tref1 =>
          Q.query(table2, "t2") { tref2 =>
            tref1.scan
              .outerEquiJoin(
                tref1.col("hfloat"),
                tref2.scan,
                tref2.col("hfloat")
              )
              .filter(tref1.col("hfloat") === 4.5)
              .done
          }
        }.interpret
          .col(1)
          .toVec == Vec(4.5)
      )

      assert(
        Q.query(table, "t1") { tref1 =>
          tref1.scan
            .aggregate(tref1.col("hint"))(
              Q.first(tref1.col("htime")) as tref1.col(
                "htime"
              ),
              Q.avg(tref1.col("hfloat")) as tref1.col(
                "hfloatavg"
              )
            )
            .done
        }.interpret
          .col("hfloatavg")
          .toVec == Vec(1.5, 2.75)
      )

      assert(
        Q.query(table, "t1") { tref1 =>
          tref1.scan
            .pivot(tref1.col("hint"), tref1.col("hbool"))(
              Q.avg(tref1.col("hfloat"))
            )
            .done
        }.interpret
          .rows(0)
          .toSTen
          .toVec == Vec(1d, 1.5, Double.NaN)
      )

      val q1 = Q.query(table, "t1") { tref1 =>
        Q.query(table2, "t2") { tref2 =>
          tref1.scan
            .innerEquiJoin(
              tref1.col("hfloat"),
              tref2.scan,
              tref2.col("hfloat")
            )
            .filter(tref1.col("hfloat") === 4.5)
            .done
        }
      }
      val q2 = RelationalAlgebra.PushDownFilters.makeChildren(q1).head
      val q3 = RelationalAlgebra.PushDownInnerJoin.makeChildren(q2).head
      assert(q3 == q1)

    }
  }
}
