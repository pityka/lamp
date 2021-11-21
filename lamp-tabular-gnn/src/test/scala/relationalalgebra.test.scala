package lamp.tgnn

import org.scalatest.funsuite.AnyFunSuite
import java.nio.channels.Channels
import Table._
import lamp._
import java.io.ByteArrayInputStream
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
      val noop = RelationalAlgebra.table(tref1).done
      assert(noop.interpret(tref1 -> table) == table)
      assert(noop.interpret(tref1 -> table) != table2)
      assert(noop.interpret(tref1 -> table2) == table2)

      val project = RelationalAlgebra.table(tref1).project(tref1.col(0),tref1.col("hbool")).project(tref1.col("hbool")).done 
      assert(project.interpret(tref1 -> table) == table.cols(3))

      val predicate = tref1.col("hbool") === 0
    
      println(predicate)
      println(predicate.negate)
      println(predicate.negate.or(predicate))

      val filter = RelationalAlgebra.table(tref1).filter(predicate.negate.or(predicate)).done 
      println(filter.interpret(tref1 -> table).stringify())
      assert(filter.interpret(tref1 -> table) equalDeep table)
      assert(filter.interpret(tref1 -> table) != table)
    }
  }
}
