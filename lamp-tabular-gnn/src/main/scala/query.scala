package lamp.tgnn

import java.util.UUID
import lamp._
import org.saddle.index.RightJoin

/* Notes:
 * - relational algebra (operator tree) != relational calculus (declarative)
 * - relational algebra: product, filter (selection), project, union, intersect, set diff. all produce tables thus algebra is closed
 * - inner join = filter of product
 * - outer join = inner join U complement of previous inner join . not associative or commutative in general
 * - identities: commutative? associative? distributive?
 * - https://www.cs.utexas.edu/~plaxton/c/337/05s/slides/RelationalDatabase-2.pdf
 * - product is binary
 * - join is binary
 * - filter is unary
 * - project is unary
 * - filter of conjunctive terms are linearizable (break them up)
 * - permutations of linearized filters are equivalent
 * - filter can be pushed below the joins if possible (other tables not needed)
 * - order of projection and filter does not matter
 * - projection can be pushed down if possible (attributes not needed)
 * - product is associative and commutative
 * - inner join is also associative and commutative  with inner join or with product
 * - union is associative and commutative
 * - group by, having, aggregation is applied after the operations in relational algebra
 * - if a join is an inner equi join on foreign key (unique) then the group by can be pushed below the joins (https://www.vldb.org/conf/1994/P354.PDF)
 * - group by is a compound group by + aggregate operation
 *
 */
object RelationalAlgebra {
  sealed trait ColumnRef
  case class IdxColumnRef(idx: Int) extends ColumnRef {
    override def toString = idx.toString
  }
  case class StringColumnRef(string: String) extends ColumnRef {
    override def toString = string
  }
  class TableRef(val alias: String) {

    override def toString = alias
    def col(i: Int): TableColumnRef = TableColumnRef(this, IdxColumnRef(i))
    def col(s: String): TableColumnRef =
      TableColumnRef(this, StringColumnRef(s))

    def asOp = TableOp(this)
    def scan = asOp
  }
  case class TableColumnRef(table: TableRef, column: ColumnRef)
      extends TableColumnRefSyntax {
    override def toString = s"${table}.$column"
  }
  case class ColumnFunctionWithOutputRef(
      function: ColumnFunction,
      outputRef: TableColumnRef
  ) {
    override def toString = function.toString
  }

  private def extractColumnRefs(table: Table): IndexedSeq[ColumnRef] =
    (0 until table.columns.size map (i =>
      IdxColumnRef(i)
    )) ++ table.nameToIndex.keySet.toSeq.map(s => StringColumnRef(s))

  def interpretBooleanExpression[S: Sc](
      factor: BooleanFactor,
      table: Table,
      mapping: Map[TableColumnRef, Int]
  ): STen = factor match {
    case ColumnFunction(columnRefs, impl) =>
      val columns = columnRefs.map(c => (c, table.columns(mapping(c)))).toMap
      impl(PredicateHelper(columns))(implicitly[Scope]).values
    case BooleanNegation(factor) =>
      val t = interpretBooleanExpression(factor, table, mapping)
      t.logicalNot
    case BooleanExpression(terms) =>
      terms
        .map { term =>
          term.factors
            .map { factor =>
              interpretBooleanExpression(factor, table, mapping)
            }
            .toList
            .reduce((a, b) => a.logicalOr(b))
        }
        .toList
        .reduce((a, b) => a.logicalAnd(b))
  }

  def interpret(root: Result, tables: Map[TableRef, Table])(implicit
      scope: Scope
  ): Table = Scope { implicit scope =>
    val sorted = topologicalSort(root).reverse

    case class Output(
        table: Table,
        columnMap: Map[TableColumnRef, Int]
    )

    def loop(ops: Seq[Op], outputs: Map[UUID, Output]): Table = {
      ops.head match {
        case op @ EquiJoin(input1, input2, col1, col2, joinType) =>
          assert(!outputs.contains(op.id))
          val inputData1 = outputs(input1.id)
          val inputData2 = outputs(input2.id)
          assert(
            inputData1.columnMap.contains(col1),
            s"Missing $col1 from table to join"
          )
          assert(
            inputData2.columnMap.contains(col2),
            s"Missing $col2 from table to join"
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

          loop(
            ops.tail,
            outputs + (op.id -> Output(joined, newMapping))
          )

        case op @ TableOp(tableRef) =>
          assert(!outputs.contains(op.id))
          val table = tables(tableRef)
          val columnRefs =
            extractColumnRefs(table).map(TableColumnRef(tableRef, _))
          val mapping = columnRefs.map {
            case ref @ TableColumnRef(_, IdxColumnRef(idx)) => (ref, idx)
            case ref @ TableColumnRef(_, StringColumnRef(string)) =>
              (ref, table.nameToIndex(string))
          }.toMap
          loop(
            ops.tail,
            outputs + (op.id -> Output(table, mapping))
          )
        case op @ Aggregate(input, groupBy, aggregations) =>
          assert(!outputs.contains(op.id))
          val inputData = outputs(input.id)
          val mapping = inputData.columnMap
          val table = inputData.table
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
          loop(
            ops.tail,
            outputs + (op.id -> Output(newTable, newColumnMap))
          )
        case op @ Pivot(input, rowKeys, colKeys, aggregate) =>
          assert(!outputs.contains(op.id))
          val inputData = outputs(input.id)
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
          loop(
            ops.tail,
            outputs + (op.id -> Output(newTable, newColumnMap))
          )
        case op @ Filter(input, booleanExpression) =>
          assert(!outputs.contains(op.id))
          val inputData = outputs(input.id)
          val newTable = Scope { implicit scope =>
            val booleanMask = interpretBooleanExpression(
              booleanExpression,
              inputData.table,
              inputData.columnMap
            )
            val indices = booleanMask.where.head
            inputData.table.rows(indices)
          }
          loop(
            ops.tail,
            outputs + (op.id -> Output(newTable, inputData.columnMap))
          )
        case op @ Product(input1, input2) =>
          assert(!outputs.contains(op.id))
          val inputData1 = outputs(input1.id)
          val inputData2 = outputs(input2.id)
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
          loop(
            ops.tail,
            outputs + (op.id -> Output(t3, newMap))
          )

        case op @ Projection(input, projectTo) =>
          assert(!outputs.contains(op.id))
          val inputData = outputs(input.id)
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
          loop(
            ops.tail,
            outputs + (op.id -> Output(projectedTable, newColumnMap))
          )
        case Result(input, _) => outputs(input.id).table
        case x => throw new RuntimeException("Unexpected op " + x)
      }
    }

    loop(sorted, Map.empty)

  }

  private def topologicalSort(root: Op): Seq[Op] = {
    type V = Op
    var order = List.empty[V]
    var marks = Set.empty[UUID]
    var currentParents = Set.empty[UUID]

    def visit(n: V): Unit =
      if (marks.contains(n.id)) ()
      else {
        if (currentParents.contains(n.id)) {
          println(s"error: loop to ${n.id}")
          ()
        } else {
          currentParents = currentParents + n.id
          val children = n.inputs
          children.foreach(visit)
          currentParents = currentParents - n.id
          marks = marks + n.id
          order = n :: order
        }
      }

    visit(root)

    order

  }

}

object Q {
  import RelationalAlgebra._
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
