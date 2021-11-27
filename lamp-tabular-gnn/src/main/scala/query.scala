package lamp.tgnn

import java.util.UUID
import lamp._

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

  case class TableWithColumnMapping(
      table: Table,
      columnMap: Map[TableColumnRef, Int]
  )
  def interpret(root: Result, tables: Map[TableRef, Table])(implicit
      scope: Scope
  ): Table = Scope { implicit scope =>
    val sorted = topologicalSort(root).reverse

    def loop(
        ops: Seq[Op],
        outputs: Map[UUID, TableWithColumnMapping]
    ): Table = {
      ops.head match {

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
            outputs + (op.id -> TableWithColumnMapping(table, mapping))
          )

        case Result(input, _) => outputs(input.id).table
        case op: Op2 =>
          assert(!outputs.contains(op.id))

          val result = op.impl(outputs(op.input1.id), outputs(op.input2.id))

          loop(
            ops.tail,
            outputs + (op.id -> result)
          )
        case op: Op1 =>
          assert(!outputs.contains(op.id))

          val result = op.impl(outputs(op.input.id))

          loop(
            ops.tail,
            outputs + (op.id -> result)
          )

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
