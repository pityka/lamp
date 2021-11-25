package lamp.tgnn
import RelationalAlgebra._
import lamp._

object Q {
  def apply(refs: TableColumnRef*)(
      impl: PredicateHelper => Scope => Table.Column
  ): ColumnFunction = ColumnFunction(refs, impl)

  def first(other: TableColumnRef) = apply(other) { input => implicit scope =>
    val col = input(other)
    Table.Column(col.values.select(0, 0).view(1), None, col.tpe, None)
  }
  def avg(other: TableColumnRef) = apply(other) { input => implicit scope =>
    val col = input(other)
    Table.Column(col.values.mean(0, true).view(1), None, col.tpe, None)
  }

  def table(alias: String): TableRef = new TableRef(alias)
  def query(table: Table, alias: String)(
      fun: TableRef => Result
  ): Result = {
    val tref = Q.table(alias)
    fun(tref).bind(tref, table)
  }
  def query(table: Table)(
      fun: TableRef => Result
  ): Result = {
    val tref = Q.table(java.util.UUID.randomUUID().toString)
    fun(tref).bind(tref, table)
  }

}
