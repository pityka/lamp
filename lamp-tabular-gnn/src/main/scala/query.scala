package lamp.tgnn

import cats.data.NonEmptyList
// import lamp.STen
import org.saddle.index.JoinType
import java.util.UUID
import lamp._
import org.saddle.index.InnerJoin
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
  }
  case class TableColumnRef(table: TableRef, column: ColumnRef) {
    override def toString = s"${table}.$column"

    def ===(other: TableColumnRef) = P(this, other) { input => implicit scope =>
      input(this).values.equ(input(other).values)
    }
    def !=(other: TableColumnRef) = P(this, other) { input => implicit scope =>
      input(this).values.equ(input(other).values).logicalNot
    }
    def <(other: TableColumnRef) = P(this, other) { input => implicit scope =>
      input(this).values.lt(input(other).values)
    }
    def >(other: TableColumnRef) = P(this, other) { input => implicit scope =>
      input(this).values.gt(input(other).values)
    }
    def <=(other: TableColumnRef) = P(this, other) { input => implicit scope =>
      input(this).values.le(input(other).values)
    }
    def >=(other: TableColumnRef) = P(this, other) { input => implicit scope =>
      input(this).values.ge(input(other).values)
    }

    def ===(other: Double) = P(this) { input => implicit scope =>
      input(this).values.equ(other)
    }
    def !=(other: Double) = P(this) { input => implicit scope =>
      input(this).values.equ(other).logicalNot
    }
    def <(other: Double) = P(this) { input => implicit scope =>
      input(this).values.lt(other)
    }
    def >(other: Double) = P(this) { input => implicit scope =>
      input(this).values.gt(other)
    }
    def <=(other: Double) = P(this) { input => implicit scope =>
      input(this).values.le(other)
    }
    def >=(other: Double) = P(this) { input => implicit scope =>
      input(this).values.ge(other)
    }

  }

  sealed trait BooleanFactor {
    def negate: BooleanFactor = BooleanNegation(this)
    def or(that: BooleanFactor*): BooleanFactor = BooleanExpression(
      NonEmptyList(BooleanTerm(NonEmptyList(this, that.toList)), Nil)
    )
    def and(that: BooleanFactor*): BooleanFactor = BooleanExpression(
      NonEmptyList(
        BooleanTerm(NonEmptyList(this, Nil)),
        that.toList.map(f => BooleanTerm(NonEmptyList(f, Nil)))
      )
    )
  }

  case class PredicateHelper(map: Map[TableColumnRef, Table.Column[_]]) {
    def apply(t: TableColumnRef) = map(t)
  }
  case class Predicate(
      columnRefs: Seq[TableColumnRef],
      impl: PredicateHelper => Scope => STen
  ) extends BooleanFactor {
    override def toString = s"[${columnRefs.mkString(",")}]"
  }
  object P {
    def apply(refs: TableColumnRef*)(
        impl: PredicateHelper => Scope => STen
    ): Predicate = Predicate(refs, impl)

  }
  case class BooleanNegation(factor: BooleanFactor) extends BooleanFactor {
    override def toString = s"\u00AC$factor"
  }
  case class BooleanTerm(factors: NonEmptyList[BooleanFactor]) { // or
    override def toString = factors.toList.mkString("(", " \u2228 ", ")")
  }
  case class BooleanExpression(terms: NonEmptyList[BooleanTerm]) // and
      extends BooleanFactor {
    override def toString = terms.toList.mkString(" \u2227 ")
  }

  case class ColumnFunction(
      columnRefs: Seq[TableColumnRef],
      impl: PredicateHelper => Scope => Table.Column[_]
  ) {
    override def toString = s"[${columnRefs.mkString(",")}]"
  }
  object F {
    def apply(refs: TableColumnRef*)(
        impl: PredicateHelper => Scope => Table.Column[_]
    ): ColumnFunction = ColumnFunction(refs, impl)

  }

  def tableRef(alias: String): TableRef = new TableRef(alias)
  def table(ref: TableRef) = TableOp(ref)
  def queryAs(table: Table, alias: String)(
      fun: TableRef => Result
  ): Result = {
    val tref = tableRef(alias)
    fun(tref).bind(tref, table)
  }

  sealed trait Op {
    val id = UUID.randomUUID()
    def inputs: Seq[Op]

    def done = Result(this)
    def project(projectTo: TableColumnRef*) = Projection(this, projectTo)
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
  case class Projection(input: Op, projectTo: Seq[TableColumnRef]) extends Op1 {
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
      RelationalAlgebra.interpret(this, tables.toMap)
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
      aggregate: Seq[(ColumnFunction, TableColumnRef)]
  ) extends Op1 {
    override def toString = s"AGGREGATE(group by ${groupBy.mkString(",")})"
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

  def extractColumnRefs(table: Table): IndexedSeq[ColumnRef] =
    (0 until table.columns.size map (i =>
      IdxColumnRef(i)
    )) ++ table.nameToIndex.keySet.toSeq.map(s => StringColumnRef(s))

  def interpretBooleanExpression[S: Sc](
      factor: BooleanFactor,
      table: Table,
      mapping: Map[TableColumnRef, Int]
  ): STen = factor match {
    case Predicate(columnRefs, impl) =>
      val columns = columnRefs.map(c => (c, table.columns(mapping(c)))).toMap
      impl(PredicateHelper(columns))(implicitly[Scope])
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
        // case op @ Aggregate(input, groupBy, aggregate) =>
          // ???
          // assert(!outputs.contains(op.id))
          // val inputData = outputs(input.id)
          // val newTable = Scope { implicit scope =>
          //   val needed
          // }
          // val newColumnMap = ???
          // loop(
          //   ops.tail,
          //   outputs + (op.id -> Output(newTable, ???))
          // )
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
          val missing = projectTo.toSet &~ providedColumns.toSet
          assert(missing.isEmpty, s"$missing columns are missing")
          val columnIdx = projectTo.map(inputData.columnMap)
          val table = inputData.table.cols(columnIdx: _*)
          val newMap = projectTo.zip(0 until projectTo.length).toMap
          loop(ops.tail, outputs + (op.id -> Output(table, newMap)))
        case Result(input, _) => outputs(input.id).table
        case x => throw new RuntimeException("Unexpected op " + x)
      }
    }

    loop(sorted, Map.empty)

  }

  def topologicalSort(root: Op): Seq[Op] = {
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

object Query {

  // case class Filter(input: Op, expression: BooleanExpression)

  sealed trait ColumnRef
  case class IdxColumnRef(idx: Int) extends ColumnRef
  case class StringColumnRef(string: String) extends ColumnRef
  class TableRef

  sealed trait TableExpression
  case class LeafTable(s: Table) extends TableExpression

  case class TableColumnRef(table: TableRef, column: ColumnRef)

  case class From(table: FromTable, joins: List[JoinedTable])
  case class FromTable(tableRef: TableExpression)
  case class JoinedTable(
      tableRef: TableExpression,
      joinType: JoinType,
      joinColumn1: TableColumnRef,
      joinColumn2: TableColumnRef
  )

  sealed trait BooleanFactor
  case class TablePredicate(
      columnRefs: Seq[TableColumnRef]
      // fun: Seq[TableColumnRef] => STen
  ) extends BooleanFactor
  case class BooleanNegation(factor: BooleanFactor) extends BooleanFactor
  case class BooleanTerm(factors: NonEmptyList[BooleanFactor]) // or
  case class BooleanExpression(terms: NonEmptyList[BooleanTerm]) // and
      extends BooleanFactor

  case class GroupBy(columnRefs: Seq[TableColumnRef])
  case class Projection(columnRefs: Seq[TableColumnRef])

  case class Query(
      from: From,
      filter: BooleanExpression,
      groupBy: GroupBy,
      projection: Projection,
      having: BooleanExpression
  ) extends TableExpression

  // def execute(
  //   query: Query,
  //   data:
  // )

}
