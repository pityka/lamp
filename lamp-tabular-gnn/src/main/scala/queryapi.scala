package lamp.tgnn
import RelationalAlgebra._
import lamp._
import org.saddle.index.InnerJoin
import org.saddle.index.JoinType
import scala.language.dynamics
object syntax {
  implicit class TableSyntax(table: Table) {
    def ref: TableRef = BoundTableRef(table)
    def query: TableOp = {
      TableOp(table.ref, Some(table))
    }

    def done = query.done
    def project(projectTo: ColumnFunctionWithOutputRef*) =
      query.project(projectTo: _*)
    def filter(expr: BooleanFactor) = query.filter(expr)
    def product(that: Op) = query.product(that)
    def union(that: Op) = query.union(that)
    def innerEquiJoin(
        thisColumn: TableColumnRef,
        that: Op,
        thatColumn: TableColumnRef
    ) = EquiJoin(query, that, thisColumn, thatColumn, InnerJoin)
    def outerEquiJoin(
        thisColumn: TableColumnRef,
        that: Op,
        thatColumn: TableColumnRef,
        joinType: JoinType = org.saddle.index.OuterJoin
    ) = EquiJoin(query, that, thisColumn, thatColumn, joinType)
    def aggregate(groupBy: TableColumnRef*)(
        aggregates: ColumnFunctionWithOutputRef*
    ) = Aggregate(query, groupBy, aggregates)
    def pivot(rowKeys: TableColumnRef, colKeys: TableColumnRef)(
        aggregate: ColumnFunction
    ) = Pivot(query, rowKeys, colKeys, aggregate)

    def asOp = query
    def scan = query
  }
}

object Q extends Dynamic {
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

  def table(alias: String): TableRef = AliasTableRef(alias)
  def query(table: Table, alias: String)(
      fun: TableOp => Op
  ): Result = {
    val tref = Q.table(alias)
    (fun(tref.scan) match {
      case x: Result => x
      case x         => x.done
    }).bind(tref, table)
  }
  def query(table: Table)(
      fun: TableRef => Result
  ): Result = {
    val tref = Q.table(java.util.UUID.randomUUID().toString)
    fun(tref).bind(tref, table)
  }

  def selectDynamic(field: String) = col(field)
  def selectDynamic(field: Int) = col(field)

  def col(i: Int): WildcardColumnRef =
    WildcardColumnRef(IdxColumnRef(i))
  def col(s: String): WildcardColumnRef =
    WildcardColumnRef(StringColumnRef(s))

}
