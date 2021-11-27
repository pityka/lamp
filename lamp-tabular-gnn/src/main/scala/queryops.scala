package lamp.tgnn

import RelationalAlgebra._

import org.saddle.index.JoinType
import java.util.UUID
import lamp._
import org.saddle.index.InnerJoin
import org.saddle.index.RightJoin

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

  def impl(input: TableWithColumnMapping)(implicit
      scope: Scope
  ): TableWithColumnMapping
}
trait Op2 extends Op {
  def input1: Op
  def input2: Op
  def inputs = List(input1, input2)
  def impl(input1: TableWithColumnMapping, input2: TableWithColumnMapping)(
      implicit scope: Scope
  ): TableWithColumnMapping
}
case class TableOp(tableRef: TableRef) extends Op0 {
  override def toString = s"TABLE($tableRef)"
}
case class Projection(input: Op, projectTo: Seq[ColumnFunctionWithOutputRef])
    extends Op1 {

  def impl(inputData: TableWithColumnMapping)(implicit
      scope: Scope
  ): TableWithColumnMapping = {
    val providedColumns = inputData.columnMap.keySet
    val neededColumns = projectTo.flatMap(_.function.columnRefs).toSet
    val missing = neededColumns &~ providedColumns.toSet
    assert(missing.isEmpty, s"$missing columns are missing")

    val inputColumnMap =
      PredicateHelper(neededColumns.toSeq.map { columnRef =>
        columnRef -> inputData.table
          .columns(inputData.columnMap(columnRef))
      }.toMap)

    val projectedTable = Table(projectTo.map {
      case ColumnFunctionWithOutputRef(
            ColumnFunction(_, impl),
            newName
          ) =>
        val aggregatedColumn =
          impl(inputColumnMap)(implicitly[Scope]).withName(
            newName.column match {
              case IdxColumnRef(_)         => None
              case StringColumnRef(string) => Some(string)
            }
          )
        aggregatedColumn
    }.toVector)

    val newColumnMap = projectTo.zipWithIndex.map {
      case (ColumnFunctionWithOutputRef(_, newName), idx) =>
        newName -> idx
    }.toMap
    TableWithColumnMapping(projectedTable, newColumnMap)
  }

  override def toString = s"PROJECT(${projectTo.mkString(",")})"
}
case class Result(input: Op, boundTables: List[(TableRef, Table)] = Nil)
    extends Op {

  def inputs = List(input)

  def bind(tableRef: TableRef, table: Table) =
    copy(boundTables = (tableRef, table) :: boundTables)
  def interpret(implicit scope: Scope): Table = 
    interpret(boundTables: _*)
  
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

  override def impl(inputData: TableWithColumnMapping)(implicit
      scope: Scope
  ): TableWithColumnMapping = {
    val newTable = Scope { implicit scope =>
      val booleanMask = interpretBooleanExpression(
        booleanExpression,
        inputData.table,
        inputData.columnMap
      )
      val indices = booleanMask.where.head
      inputData.table.rows(indices)
    }
    TableWithColumnMapping(newTable, inputData.columnMap)

  }

}
case class Aggregate(
    input: Op,
    groupBy: Seq[TableColumnRef],
    aggregate: Seq[ColumnFunctionWithOutputRef]
) extends Op1 {
  override def toString = s"AGGREGATE(group by ${groupBy.mkString(",")})"

  def impl(
      inputData: TableWithColumnMapping
  )(implicit scope: Scope): TableWithColumnMapping = {
    val mapping = inputData.columnMap
    val table = inputData.table
    val aggregations = aggregate
    val newTable = Scope { implicit scope =>
      val groupByColumnIndices = groupBy.map(mapping)
      val locations = table.groupByGroupIndices(groupByColumnIndices: _*)
      val allNeededColumnRefs =
        aggregations.flatMap(_.function.columnRefs).distinct

      val aggregates = locations
        .map { locs =>
          val selectedColumns = allNeededColumnRefs.map { columnRef =>
            columnRef -> table
              .columns(mapping(columnRef))
              .select(locs)
          }.toMap
          Table(aggregations.map {
            case ColumnFunctionWithOutputRef(
                  ColumnFunction(_, impl),
                  newName
                ) =>
              val aggregatedColumn =
                impl(PredicateHelper(selectedColumns))(
                  implicitly[Scope]
                ).withName(newName.column match {
                  case IdxColumnRef(_)         => None
                  case StringColumnRef(string) => Some(string)
                })
              aggregatedColumn
          }.toVector)

        }

      Table.union(aggregates: _*)

    }
    val newColumnMap = aggregations.zipWithIndex.map {
      case (ColumnFunctionWithOutputRef(_, newName), idx) =>
        newName -> idx
    }.toMap
    TableWithColumnMapping(newTable, newColumnMap)

  }

}
case class Pivot(
    input: Op,
    rowKeys: TableColumnRef,
    colKeys: TableColumnRef,
    aggregate: ColumnFunction
) extends Op1 {
  override def toString = s"PIVOT($rowKeys x $colKeys)"

  def impl(inputData: TableWithColumnMapping)(implicit
      scope: Scope
  ): TableWithColumnMapping = {
    val mapping = inputData.columnMap
    val table = inputData.table
    val allNeededColumnRefs =
      aggregate.columnRefs.distinct
    val newTable = Scope { implicit scope =>
      val colIdx0 = mapping(rowKeys)
      val colIdx1 = mapping(colKeys)
      table.pivot(colIdx0, colIdx1)(table =>
        Table(
          Vector(aggregate.impl(PredicateHelper(allNeededColumnRefs.map {
            columnRef =>
              (columnRef, table.columns(mapping(columnRef)))
          }.toMap))(implicitly[Scope]))
        )
      )

    }

    val newColumnMap = {
      val aggregateRef = aggregate.columnRefs.head
      val colnames = newTable.columns
        .drop(1)
        .zipWithIndex
        .map(v => v._1.name.getOrElse(v._2.toString))
      Map(rowKeys -> 0) ++ colnames.zipWithIndex.map { case (name, idx) =>
        (
          aggregateRef.table.col(
            aggregateRef.column.toString + "." + name
          ),
          idx + 1
        )
      }
    }
    TableWithColumnMapping(newTable, newColumnMap)
  }

}

case class Product(input1: Op, input2: Op) extends Op2 {
  override def toString = "PRODUCT"

  def impl(
      inputData1: TableWithColumnMapping,
      inputData2: TableWithColumnMapping
  )(implicit scope: Scope) = {

    val inputTableRefs1 = inputData1.columnMap.keys.map(_.table).toSet
    val inputTableRefs2 = inputData2.columnMap.keys.map(_.table).toSet
    assert(
      (inputTableRefs1 & inputTableRefs2).isEmpty,
      "Joining same tables aliases"
    )
    val t3 = Scope { implicit scope =>
      val idx1 = STen.arange_l(
        0,
        inputData1.table.numRows,
        1L,
        inputData1.table.device.to(STen.lOptions)
      )
      val idx2 = STen.arange_l(
        0,
        inputData2.table.numRows,
        1L,
        inputData2.table.device.to(STen.lOptions)
      )
      val cartes = STen.cartesianProduct(List(idx1, idx2))
      val t1 = inputData1.table.rows(cartes.select(1, 0))
      val t2 = inputData2.table.rows(cartes.select(1, 1))
      t1.bind(t2)
    }
    val newMap = inputData1.columnMap ++ inputData2.columnMap.map(v =>
      (v._1, v._2 + inputData1.table.numCols)
    )
    TableWithColumnMapping(t3, newMap)
  }
}

case class EquiJoin(
    input1: Op,
    input2: Op,
    column1: TableColumnRef,
    column2: TableColumnRef,
    joinType: JoinType
) extends Op2 {
  override def toString = s"EQUIJOIN($column1,$column2,$joinType)"

  def impl(
      inputData1: TableWithColumnMapping,
      inputData2: TableWithColumnMapping
  )(implicit scope: Scope): TableWithColumnMapping = {
    val col1 = column1
    val col2 = column2
    assert(
      inputData1.columnMap.contains(column1),
      s"Missing $column1 from table to join"
    )
    assert(
      inputData2.columnMap.contains(column2),
      s"Missing $column2 from table to join"
    )
    val joined = inputData1.table.join(
      inputData1.columnMap(col1),
      inputData2.table,
      inputData2.columnMap(col2),
      joinType
    )
    val oldCol2Idx = inputData2.columnMap(col2)
    val oldCol1Idx = inputData1.columnMap(col1)

    val leftMapping = inputData1.columnMap
      .map { v =>
        val shiftback =
          if (v._2 > oldCol1Idx && joinType == RightJoin) -1
          else 0
        (v._1, v._2 + shiftback)
      }
      .filterNot { case (ref, _) => joinType == RightJoin && ref == col1 }
    val rightMapping = inputData2.columnMap
      .map { v =>
        val shiftback =
          if (v._2 > oldCol2Idx && joinType != RightJoin) -1
          else 0
        (v._1, v._2 + inputData1.table.numCols + shiftback)
      }
      .filterNot { case (ref, _) => joinType != RightJoin && ref == col2 }
    val newMapping = leftMapping ++ rightMapping
    TableWithColumnMapping(joined, newMapping)
  }

}
