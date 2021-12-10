package lamp.tgnn

import RelationalAlgebra._

import org.saddle.index.JoinType
import java.util.UUID
import lamp._
import org.saddle.index.InnerJoin
import org.saddle.index.RightJoin
import org.saddle.index.LeftJoin
import org.saddle.index.OuterJoin
import scala.language.dynamics

case class InputWithNeededColumns(op: Op, neededColumns: Seq[TableColumnRef])

sealed trait Op {
  val id = UUID.randomUUID()
  def inputs: Seq[InputWithNeededColumns]

  def replace(old: UUID, n: Op): Op

  def done = Result(this)
  def project(projectTo: ColumnFunctionWithOutputRef*) =
    Projection(this, projectTo)
  def filter(expr: BooleanFactor) = Filter(this, expr)
  def product(that: Op) = Product(this, that)
  def union(that: Op) = Union(this, that)
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
    done.bind(tables: _*).check.optimize().interpret
  def result(implicit scope: Scope) =
    doneAndInterpret()

}
trait Op0 extends Op {
  def inputs = Nil
}
trait Op1 extends Op {
  def input: Op
  def neededColumns: Seq[TableColumnRef]
  def inputs = List(InputWithNeededColumns(input, neededColumns))

  def providesColumns(input: ColumnSet): Either[String, ColumnSet]

  def estimate(input: TableEstimate): TableEstimate
  def impl(input: TableWithColumnMapping)(implicit
      scope: Scope
  ): TableWithColumnMapping
}
trait Op2 extends Op {
  def input1: Op
  def input2: Op
  def neededColumns1: Seq[TableColumnRef]
  def neededColumns2: Seq[TableColumnRef]
  def inputs = List(
    InputWithNeededColumns(input1, neededColumns1),
    InputWithNeededColumns(input2, neededColumns2)
  )

  def providesColumns(
      input1: ColumnSet,
      input2: ColumnSet
  ): Either[String, ColumnSet]

  def estimate(input1: TableEstimate, input2: TableEstimate): TableEstimate

  def impl(input1: TableWithColumnMapping, input2: TableWithColumnMapping)(
      implicit scope: Scope
  ): TableWithColumnMapping
}
case class TableOp(tableRef: TableRef, boundTable: Option[Table])
    extends Op0
    with Dynamic {
  override def toString = s"TABLE($tableRef)"
  def neededColumns = Nil

  def replace(old: UUID, n: Op) = this

  def col(col: String) = tableRef.col(col)
  def col(col: Int) = tableRef.col(col)
  def selectDynamic(field: String) = col(field)
  def selectDynamic(field: Int) = col(field)

}
case class Projection(input: Op, projectTo: Seq[ColumnFunctionWithOutputRef])
    extends Op1 {

  def estimate(input: TableEstimate): TableEstimate =
    input.copy(columns = projectTo.size.toLong)

  def replace(old: UUID, n: Op) = {
    if (input.id == old) copy(input = n)
    else copy(input = input.replace(old, n))
  }

  val neededColumns = projectTo.flatMap(_.function.columnRefs).distinct

  def providesColumns(input: ColumnSet) = {
    if (neededColumns.forall(n => input.hasMatchingColumn(n)))
      Right(
        ColumnSet(
          neededColumns.map(n => input.getMatchingQualifiedColumnRef(n)).toSet
        )
      )
    else
      Left(
        s"Missing column ${neededColumns.filterNot(n => input.hasMatchingColumn(n))}"
      )
  }

  def impl(inputData: TableWithColumnMapping)(implicit
      scope: Scope
  ): TableWithColumnMapping = {
    val providedColumns = inputData.providedColumns
    val neededColumns = projectTo.flatMap(_.function.columnRefs).toSet
    val missing = neededColumns &~ providedColumns.toSet
    assert(missing.isEmpty, s"$missing columns are missing")

    val inputColumnMap =
      PredicateHelper(neededColumns.toSeq.map { columnRef =>
        columnRef -> inputData.table
          .columns(inputData.getMatchingColumnIdx(columnRef))
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
case class Result(input: Op, boundTables: Map[TableRef, Table] = Map.empty)
    extends Op {

  override def done = this

  def check = {
    val sorted = RelationalAlgebra.topologicalSort(this).reverse
    val errors = providedReferences(sorted, boundTables)
    if (errors.isLeft) throw new RuntimeException(errors.left.toOption.get)
    this
  }

  def replace(old: UUID, n: Op) = {
    if (input.id == old) copy(input = n)
    else copy(input = input.replace(old, n))
  }

  def inputs = List(InputWithNeededColumns(input, Nil))

  def neededColumns = Nil

  def bind(tableRef: TableRef, table: Table) =
    copy(boundTables = (boundTables - tableRef) + ((tableRef, table)))
  def bind(tables: (TableRef, Table)*) =
    copy(boundTables = (boundTables -- tables.map(_._1)) ++ tables)

  def interpret(implicit
      scope: Scope
  ): Table =
    RelationalAlgebra.interpret(this)

  override def toString = "RESULT"
  def stringify: String = {

    def loop(head: Op, indent: Int): Vector[String] = {
      val inputs = head.inputs.toVector
      val me = head.toString
      val kids = inputs.flatMap(k => loop(k.op, indent + 1))
      val line = (0 until indent map (_ => "  ") mkString) + me
      kids.prepended(line)
    }

    loop(this, 0).mkString("\n")

  }
  def optimize(depth: Int = 10): Result = {
    val candidates =
      RelationalAlgebra.depthFirstSearch(this, Mutation.list, depth)
    candidates.minBy(c => RelationalAlgebra.estimate(c))
  }
}
case class Filter(input: Op, booleanExpression: BooleanFactor) extends Op1 {
  override def toString = s"FILTER($booleanExpression)"

  def replace(old: UUID, n: Op) = {
    if (input.id == old) copy(input = n)
    else copy(input = input.replace(old, n))
  }
  def estimate(input: TableEstimate): TableEstimate = input
  def neededColumns = columnsReferencedByBooleanExpression(booleanExpression)

  def providesColumns(input: ColumnSet) = {
    val expressionOK = verifyBooleanExpression(
      booleanExpression,
      input
    )
    if (neededColumns.forall(n => input.hasMatchingColumn(n)) && expressionOK)
      Right(
        input
      )
    else
      Left(
        s"In $this: Missing or ambiguous column(s) ${neededColumns
          .filterNot(n => input.hasMatchingColumn(n))
          .mkString("{", ", ", "}")} ${input.providedColumns.mkString("{", ", ", "}")}"
      )
  }

  override def impl(inputData: TableWithColumnMapping)(implicit
      scope: Scope
  ): TableWithColumnMapping = {
    val newTable = Scope { implicit scope =>
      val booleanMask = interpretBooleanExpression(
        booleanExpression,
        inputData
      )
      val indices = booleanMask.where.head
      inputData.table.rows(indices)
    }
    inputData.copy(table = newTable)

  }

}
case class Aggregate(
    input: Op,
    groupBy: Seq[TableColumnRef],
    aggregate: Seq[ColumnFunctionWithOutputRef]
) extends Op1 {
  override def toString = s"AGGREGATE(group by ${groupBy.mkString(",")})"

  def replace(old: UUID, n: Op) = {
    if (input.id == old) copy(input = n)
    else copy(input = input.replace(old, n))
  }

  def estimate(input: TableEstimate): TableEstimate = input

  def neededColumns =
    (groupBy ++ aggregate.flatMap(_.function.columnRefs)).distinct

  def providesColumns(input: ColumnSet) =
    // input ++ aggregate.map(_.outputRef)
    {
      if (neededColumns.forall(n => input.hasMatchingColumn(n)))
        Right(
          ColumnSet(
            input.providedColumns ++ aggregate.map(_.outputRef)
          )
        )
      else
        Left(
          s"Missing column ${neededColumns.filterNot(n => input.hasMatchingColumn(n))}"
        )
    }

  def impl(
      inputData: TableWithColumnMapping
  )(implicit scope: Scope): TableWithColumnMapping = {
    val groupByColumnIndices = groupBy.map(inputData.getMatchingColumnIdx)
    val table = inputData.table
    val aggregations = aggregate
    val newTable = Scope { implicit scope =>
      val locations = table.groupByGroupIndices(groupByColumnIndices: _*)
      val allNeededColumnRefs =
        aggregations.flatMap(_.function.columnRefs).distinct

      val aggregates = locations
        .map { locs =>
          val selectedColumns = allNeededColumnRefs.map { columnRef =>
            columnRef -> table
              .columns(inputData.getMatchingColumnIdx(columnRef))
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

  def estimate(input: TableEstimate): TableEstimate =
    TableEstimate(input.rows, input.rows)

  def replace(old: UUID, n: Op) = {
    if (input.id == old) copy(input = n)
    else copy(input = input.replace(old, n))
  }

  def neededColumns =
    (rowKeys +: colKeys +: aggregate.columnRefs).distinct
  def providesColumns(input: ColumnSet): Either[String, ColumnSet] = {
    if (neededColumns.forall(n => input.hasMatchingColumn(n)))
      Right(
        ColumnSet(
          Set(input.getMatchingQualifiedColumnRef(rowKeys))
        )
      )
    else
      Left(
        s"Missing column ${neededColumns.filterNot(n => input.hasMatchingColumn(n))}"
      )
  }

  def impl(inputData: TableWithColumnMapping)(implicit
      scope: Scope
  ): TableWithColumnMapping = {
    val table = inputData.table
    val allNeededColumnRefs =
      aggregate.columnRefs.distinct
    val newTable = Scope { implicit scope =>
      val colIdx0 = inputData.getMatchingColumnIdx(rowKeys)
      val colIdx1 = inputData.getMatchingColumnIdx(colKeys)
      table.pivot(colIdx0, colIdx1)(table =>
        Table(
          Vector(
            aggregate.impl(PredicateHelper(allNeededColumnRefs.map {
              columnRef =>
                (
                  columnRef,
                  table.columns(inputData.getMatchingColumnIdx(columnRef))
                )
            }.toMap))(implicitly[Scope])
          )
        )
      )

    }

    val newColumnMap = {
      val tableRef = inputData.providedColumns.head.table

      val colnames = newTable.columns
        .drop(1)
        .zipWithIndex
        .map(v => v._1.name.getOrElse(v._2.toString))
      Map(
        inputData.getMatchingQualifiedColumnRef(rowKeys) -> 0
      ) ++ colnames.zipWithIndex.map { case (name, idx) =>
        (
          tableRef.col(
            colKeys.column.toString + "." + name
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

  def replace(old: UUID, n: Op) = {
    if (input1.id == old) copy(input1 = n, input2 = input2.replace(old, n))
    else if (input2.id == old) copy(input2 = n, input1 = input1.replace(old, n))
    else copy(input1 = input1.replace(old, n), input2 = input2.replace(old, n))

  }

  def estimate(input1: TableEstimate, input2: TableEstimate): TableEstimate =
    TableEstimate(input1.rows * input2.rows, input1.columns + input2.columns)

  def neededColumns1 = Nil
  def neededColumns2 = Nil
  def providesColumns(
      input1: ColumnSet,
      input2: ColumnSet
  ) =
    Right(
      ColumnSet(
        input1.providedColumns ++ input2.providedColumns
      )
    )

  def impl(
      inputData1: TableWithColumnMapping,
      inputData2: TableWithColumnMapping
  )(implicit scope: Scope) = {

    val inputTableRefs1 = inputData1.providedColumns.map(_.table)
    val inputTableRefs2 = inputData2.providedColumns.map(_.table)
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
    val newMap =
      inputData1.mergeMaps(inputData2.shiftColumnsUp(inputData1.table.numCols))
    TableWithColumnMapping(t3, newMap)
  }
}

case class Union(input1: Op, input2: Op) extends Op2 {
  override def toString = "UNION"

  def replace(old: UUID, n: Op) = {
    if (input1.id == old) copy(input1 = n, input2 = input2.replace(old, n))
    else if (input2.id == old) copy(input2 = n, input1 = input1.replace(old, n))
    else copy(input1 = input1.replace(old, n), input2 = input2.replace(old, n))

  }

  def estimate(input1: TableEstimate, input2: TableEstimate): TableEstimate =
    TableEstimate(input1.rows + input2.rows, input1.columns)

  def neededColumns1 = Nil
  def neededColumns2 = Nil
  def providesColumns(
      input1: ColumnSet,
      input2: ColumnSet
  ) =
    if (input1 == input2)
      Right(
        ColumnSet(
          input1.providedColumns
        )
      )
    else Left(s"Can't union different column sets")

  def impl(
      inputData1: TableWithColumnMapping,
      inputData2: TableWithColumnMapping
  )(implicit scope: Scope) = {

    assert(inputData1.providedColumns == inputData2.providedColumns)

    val columnOrder = inputData1.columnOrder

    val t3 = Scope { implicit scope =>
      val t1 = inputData1.table
      val t2 = inputData2.table.cols(
        columnOrder.map(inputData2.getMatchingColumnIdx): _*
      )
      t1.union(t2)
    }
    val newMap =
      inputData1 mergeMaps inputData2.shiftColumnsUp(inputData1.table.numCols)
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

  def estimate(input1: TableEstimate, input2: TableEstimate): TableEstimate =
    joinType match {
      case InnerJoin =>
        TableEstimate(
          math.min(input1.rows, input2.rows),
          input1.columns + input2.columns
        )
      case LeftJoin =>
        TableEstimate(input1.rows, input1.columns + input2.columns)
      case RightJoin =>
        TableEstimate(input2.rows, input1.columns + input2.columns)
      case OuterJoin =>
        TableEstimate(
          input1.rows + input2.rows,
          input1.columns + input2.columns
        )
    }

  def neededColumns1 = Vector(column1)
  def neededColumns2 = Vector(column2)

  def replace(old: UUID, n: Op) = {
    if (input1.id == old) copy(input1 = n, input2 = input2.replace(old, n))
    else if (input2.id == old) copy(input2 = n, input1 = input1.replace(old, n))
    else copy(input1 = input1.replace(old, n), input2 = input2.replace(old, n))

  }
  def providesColumns(
      input1: ColumnSet,
      input2: ColumnSet
  ) =
    if (input1.hasMatchingColumn(column1) && input2.hasMatchingColumn(column2))
      Right(
        ColumnSet(
          input1.providedColumns ++ input2.providedColumns
        )
      )
    else Left(s"Can't union different column sets")

  def impl(
      inputData1: TableWithColumnMapping,
      inputData2: TableWithColumnMapping
  )(implicit scope: Scope): TableWithColumnMapping = {
    val col1 = column1
    val col2 = column2
    assert(
      inputData1.hasMatchingColumn(column1),
      s"Missing $column1 from table to join"
    )
    assert(
      inputData2.hasMatchingColumn(column2),
      s"Missing $column2 from table to join"
    )
    val joined = inputData1.table.join(
      inputData1.getMatchingColumnIdx(col1),
      inputData2.table,
      inputData2.getMatchingColumnIdx(col2),
      joinType
    )
    val oldCol2Idx = inputData2.getMatchingColumnIdx(col2)
    val oldCol1Idx = inputData1.getMatchingColumnIdx(col1)

    val leftMapping = inputData1.getColumnMapping
      .map { v =>
        val shiftback =
          if (v._2 > oldCol1Idx && joinType == RightJoin) -1
          else 0
        (v._1, v._2 + shiftback)
      }
      .filterNot { case (ref, _) =>
        joinType == RightJoin && (ref matches col1)
      }
    val rightMapping = inputData2.getColumnMapping
      .map { v =>
        val shiftback =
          if (v._2 > oldCol2Idx && joinType != RightJoin) -1
          else 0
        (v._1, v._2 + inputData1.table.numCols + shiftback)
      }
      .filterNot { case (ref, _) =>
        joinType != RightJoin && (ref matches col2)
      }
    val newMapping = leftMapping ++ rightMapping
    TableWithColumnMapping(joined, newMapping.toMap)
  }

}
