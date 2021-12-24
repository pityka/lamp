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
            tref1
              .product(
                tref2
              )
              .done
          }
        }.doneAndInterpret()
          .col(6)
          .toVec == Vec(5.5, 4.5, 6.0, 5.5, 4.5, 6.0, 5.5, 4.5, 6.0)
        // .toMat == Mat(Vec(4.5))
      )

      assert(
        Q.query(table, "t1") { tref1 =>
          Q.query(table2, "t2") { tref2 =>
            tref1
              .outerEquiJoin(
                tref1.col("hfloat"),
                tref2,
                tref2.col("hfloat")
              )
              .filter(tref1.col("hfloat") === 4.5)
              .done
          }
        }.interpret
          .col(1)
          .toVec == Vec(4.5)
      )
      import lamp.tgnn.syntax._

      assert(
        table
          .innerEquiJoin(
            Q.hint,
            table2.query,
            Q.col("hint")
          )
          .filter(table.ref.hfloat === 1.5)
          .result
          .col(1)
          .toVec == Vec(1.5)
      )

      assert(
        Q.query(table, "t1") { tref1 =>
          tref1
            .aggregate(tref1.col("hint"))(
              Q.first(tref1.col("htime")) as tref1.col(
                "htime"
              ),
              Q.avg(tref1.col("hfloat")) as tref1.col(
                "hfloatavg"
              )
            )
        }.interpret
          .col("hfloatavg")
          .toVec == Vec(1.5, 2.75)
      )

      assert(
        Q.query(table, "t1") { tref1 =>
          tref1
            .pivot(tref1.col("hint"), tref1.col("hbool"))(
              Q.avg(tref1.col("hfloat"))
            )
        }.doneAndInterpret()
          .rows(0)
          .toSTen
          .toVec == Vec(1d, 1.5, Double.NaN)
      )

      val q1 = Q.query(table, "t1") { tref1 =>
        Q.query(table2, "t2") { tref2 =>
          tref1
            .innerEquiJoin(
              tref1.hint,
              tref2,
              tref2.hint
            )
            .filter(tref1.hfloat === 1.5)
            .project(tref1.htext.self)
            .union(tref1.project(tref1.htext.self))
        }
      }
      val q2 = PushDownFilters.makeChildren(q1).toOption.get.head
      val q3 = PushDownInnerJoin.makeChildren(q2).toOption.get.head
      assert(q3 == q1)
      assert(q2 != q1)

      assert(q1.optimize() == q2)

      assert(q1.optimize().interpret equalDeep q1.interpret)

      assert(
        table
          .innerEquiJoin(
            Q.hint,
            table2.query
              .filter(Q.col("hfloat") >= 0d and Q.col("hfloat") > 0)
              .project(
                table2.ref.col("hfloat").self,
                table2.ref.col("hint").self
              ),
            Q.col("hint")
          )
          .project(table.ref.col("hfloat").self)
          .result
          .col(0)
          .toVec == Vec(1.5, 2.5, 2.5, 3.0, 3.0)
      )
      assert(
        table
          .innerEquiJoin(
            Q.hint,
            table2.query,
            Q.col("hint")
          )
          .innerEquiJoin(Q.hint, table.query, Q.hint)
          .aggregate(Q.hint)(
            Q.avg(table.ref.hfloat).as(Q.table("aggr").col("boo"))
          )
          .result
          .col(0)
          .toVec == Vec(1.5, 2.75)
      )
    }
  }
}
