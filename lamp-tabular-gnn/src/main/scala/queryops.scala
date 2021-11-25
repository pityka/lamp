package lamp.tgnn

import RelationalAlgebra._

import org.saddle.index.JoinType
import java.util.UUID
import lamp._
import org.saddle.index.InnerJoin

sealed trait Op {
  val id = UUID.randomUUID()
  def inputs: Seq[Op]

  def done = Result(this)
  def project(projectTo: ColumnFunctionWithOutputRef*) =
    Projection(this, projectTo)
  def filter(expr: BooleanFactor) = Filter(this, expr)
  def product(that: Op) = Product(this, that)
  def innerEquiJoin(
      thisColumn: TableColumnRef,
      that: Op,
      thatColumn: TableColumnRef
  ) = EquiJoin(this, that, thisColumn, thatColumn, InnerJoin)
  def outerEquiJoin(
      thisColumn: TableColumnRef,
      that: Op,
      thatColumn: TableColumnRef,
      joinType: JoinType = org.saddle.index.OuterJoin
  ) = EquiJoin(this, that, thisColumn, thatColumn, joinType)
  def aggregate(groupBy: TableColumnRef*)(
      aggregates: ColumnFunctionWithOutputRef*
  ) = Aggregate(this, groupBy, aggregates)
  def pivot(rowKeys: TableColumnRef, colKeys: TableColumnRef)(
      aggregate: ColumnFunction
  ) = Pivot(this, rowKeys, colKeys, aggregate)

  def doneAndInterpret(tables: (TableRef, Table)*)(implicit scope: Scope) =
    done.interpret(tables: _*)

}
trait Op0 extends Op {
  def inputs = Nil
}
trait Op1 extends Op {
  def input: Op
  def inputs = List(input)
}
trait Op2 extends Op {
  def input1: Op
  def input2: Op
  def inputs = List(input1, input2)
}
case class TableOp(tableRef: TableRef) extends Op0 {
  override def toString = s"TABLE($tableRef)"
}
case class Projection(input: Op, projectTo: Seq[ColumnFunctionWithOutputRef])
    extends Op1 {
  override def toString = s"PROJECT(${projectTo.mkString(",")})"
}
case class Result(input: Op, boundTables: List[(TableRef, Table)] = Nil)
    extends Op1 {
  def bind(tableRef: TableRef, table: Table) =
    copy(boundTables = (tableRef, table) :: boundTables)
  def interpret(implicit scope: Scope): Table = {
    interpret(boundTables: _*)
  }
  def interpret(tables: (TableRef, Table)*)(implicit
      scope: Scope
  ): Table = {
    assert(
      tables.map(_._1).distinct.size == tables.size,
      "Non unique table refs"
    )
    RelationalAlgebra.interpret(
      this,
      (boundTables.toMap -- tables.map(_._1)) ++ tables
    )
  }
  override def toString = "RESULT"
  def stringify: String = {

    def loop(head: Op, indent: Int): Vector[String] = {
      val inputs = head.inputs.toVector
      val me = head.toString
      val kids = inputs.flatMap(k => loop(k, indent + 1))
      val line = (0 until indent map (_ => "  ") mkString) + me
      kids.prepended(line)
    }

    loop(this, 0).mkString("\n")

  }
}
case class Filter(input: Op, booleanExpression: BooleanFactor) extends Op1 {
  override def toString = s"FILTER($booleanExpression)"
}
case class Aggregate(
    input: Op,
    groupBy: Seq[TableColumnRef],
    aggregate: Seq[ColumnFunctionWithOutputRef]
) extends Op1 {
  override def toString = s"AGGREGATE(group by ${groupBy.mkString(",")})"
}
case class Pivot(
    input: Op,
    rowKeys: TableColumnRef,
    colKeys: TableColumnRef,
    aggregate: ColumnFunction
) extends Op1 {
  override def toString = s"PIVOT($rowKeys x $colKeys)"
}

case class Product(input1: Op, input2: Op) extends Op2 {
  override def toString = "PRODUCT"
}

case class EquiJoin(
    input1: Op,
    input2: Op,
    column1: TableColumnRef,
    column2: TableColumnRef,
    joinType: JoinType
) extends Op2 {
  override def toString = s"EQUIJOIN($column1,$column2,$joinType)"
}
