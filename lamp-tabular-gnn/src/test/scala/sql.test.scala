package lamp.tgnn

import org.scalatest.funsuite.AnyFunSuite

class SQLSuite extends AnyFunSuite {
  test("dfs"){
    println(Sql.query.parse("select * from table1 inner join table2 on table.id = table.id and ( table.id = table.id or table.id = table.id ) and table.id = table.id"))
  }
}
