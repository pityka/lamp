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

      val tref1 = RelationalAlgebra.tableRef("t1")
      val tref2 = RelationalAlgebra.tableRef("t2")
      val noop = RelationalAlgebra.table(tref1).done
      assert(noop.interpret(tref1 -> table) == table)
      assert(noop.interpret(tref1 -> table) != table2)
      assert(noop.interpret(tref1 -> table2) == table2)

      val project = RelationalAlgebra
        .table(tref1)
        .project(tref1.col(0).select.asSelf, tref1.col("hbool").select.asSelf)
        .project(tref1.col("hbool").select.asSelf)
        .done
      assert(project.interpret(tref1 -> table) == table.cols(3))

      val predicate = tref1.col("hbool") === 0

      val filter = RelationalAlgebra
        .table(tref1)
        .filter(predicate.negate.or(predicate))
        .done
      assert(filter.interpret(tref1 -> table) equalDeep table)
      assert(filter.interpret(tref1 -> table) != table)

      val innerJoin =
        RelationalAlgebra
          .table(tref1)
          .innerEquiJoin(
            tref1.col("hint"),
            RelationalAlgebra.table(tref2),
            tref2.col("hint")
          )
          .done

      assert(
        innerJoin.interpret(tref1 -> table, tref2 -> table2) equalDeep table
          .join(0, table2, 0)
      )

      assert(
        RelationalAlgebra
          .queryAs(table, "t1") { tref1 =>
            RelationalAlgebra.queryAs(table2, "t2") { tref2 =>
              tref1.asOp
                .innerEquiJoin(
                  tref1.col("hint"),
                  tref2.asOp,
                  tref2.col("hint")
                )
                .filter(tref2.col("hfloat") === 4.5)
                .done
            }
          }
          .interpret
          .col(5).values
          .toMat == Mat(Vec(4.5))
      )
      assert(
        RelationalAlgebra
          .queryAs(table, "t1") { tref1 =>
            RelationalAlgebra.queryAs(table2, "t2") { tref2 =>
              tref1.asOp
                .product(
                  tref2.asOp
                )
                .done
            }
          }
          .interpret
          .col(6)
          .toVec == Vec(5.5, 4.5, 6.0, 5.5, 4.5, 6.0, 5.5, 4.5, 6.0)
        // .toMat == Mat(Vec(4.5))
      )

      assert(
        RelationalAlgebra
          .queryAs(table, "t1") { tref1 =>
            RelationalAlgebra.queryAs(table2, "t2") { tref2 =>
              tref1.asOp
                .outerEquiJoin(
                  tref1.col("hfloat"),
                  tref2.asOp,
                  tref2.col("hfloat")
                )
                .filter(tref1.col("hfloat") === 4.5)
                .done
            }
          }
          .interpret
          .col(1)
          .toVec == Vec(4.5)
        // .col(5)
        // .toMat == Mat(Vec(4.5))
      )
      
        assert(RelationalAlgebra
          .queryAs(table, "t1") { tref1 =>
            val q = tref1.asOp.aggregate(tref1.col("hint"))(
              RelationalAlgebra.P.first(tref1.col("htime")) as tref1.col("htime"),
              RelationalAlgebra.P.avg(tref1.col("hfloat")) as tref1.col("hfloatavg"),
            ).done
            println(q.stringify)
            q
          }
          .interpret.col("hfloatavg").toVec == Vec(1.5,2.75))
        // .col(5)
        // .toMat == Mat(Vec(4.5))
      
    }
  }
}
