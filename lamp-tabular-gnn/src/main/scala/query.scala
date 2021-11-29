package lamp.tgnn

import java.util.UUID
import lamp._
import scala.collection.immutable.Queue
import org.saddle.index.InnerJoin

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
  def columnsReferencedByBooleanExpression(
      factor: BooleanFactor
  ): Seq[TableColumnRef] = factor match {
    case ColumnFunction(columnRefs, _) =>
      columnRefs.distinct
    case BooleanNegation(factor) =>
      columnsReferencedByBooleanExpression(factor)
    case BooleanExpression(terms) =>
      terms.toList
        .flatMap { term =>
          term.factors.toList.flatMap { factor =>
            columnsReferencedByBooleanExpression(factor)
          }.toList

        }

  }

  case class TableWithColumnMapping(
      table: Table,
      columnMap: Map[TableColumnRef, Int]
  )

  def interpret(root: Result)(implicit
      scope: Scope
  ): Table = Scope { implicit scope =>
    val sorted = topologicalSort(root).reverse

    val tables = root.boundTables.toMap

    def makeMapping(tableRef: TableRef): TableWithColumnMapping = {
      val table = tables(tableRef)
      val columnRefs =
        extractColumnRefs(table).map(TableColumnRef(tableRef, _))
      val mapping = columnRefs.map {
        case ref @ TableColumnRef(_, IdxColumnRef(idx)) => (ref, idx)
        case ref @ TableColumnRef(_, StringColumnRef(string)) =>
          (ref, table.nameToIndex(string))
      }.toMap
      TableWithColumnMapping(table, mapping)
    }

    def loop(
        ops: Seq[Op],
        outputs: Map[UUID, TableWithColumnMapping]
    ): Table = {
      ops.head match {

        case op @ TableOp(tableRef) =>
          assert(!outputs.contains(op.id))
          loop(
            ops.tail,
            outputs + (op.id -> makeMapping(tableRef))
          )

        case op @ Result(input, _) =>
          assert(!outputs.contains(op.id))
          outputs(input.id).table
        case op: Op2 =>
          assert(!outputs.contains(op.id))

          loop(
            ops.tail,
            outputs + (op.id -> op
              .impl(outputs(op.input1.id), outputs(op.input2.id)))
          )
        case op: Op1 =>
          assert(!outputs.contains(op.id))

          loop(
            ops.tail,
            outputs + (op.id -> op.impl(outputs(op.input.id)))
          )

        case x => throw new RuntimeException("Unexpected op " + x)
      }
    }

    loop(sorted, Map.empty)

  }
  def providedReferences(topoSorted: Seq[Op], tables: Map[TableRef, Table]) = {

    def extractColumnRefsFromTableRef(
        tableRef: TableRef
    ): Seq[TableColumnRef] = {
      val table = tables(tableRef)
      val columnRefs =
        extractColumnRefs(table).map(TableColumnRef(tableRef, _))

      columnRefs
    }

    def loop(
        ops: Seq[Op],
        outputs: Map[UUID, Seq[TableColumnRef]]
    ): Map[UUID, Seq[TableColumnRef]] = {
      ops.head match {

        case op @ TableOp(tableRef) =>
          assert(!outputs.contains(op.id))
          loop(
            ops.tail,
            outputs + (op.id -> extractColumnRefsFromTableRef(tableRef))
          )

        case op @ Result(_, _) =>
          assert(!outputs.contains(op.id))
          outputs
        case op: Op2 =>
          assert(!outputs.contains(op.id))

          loop(
            ops.tail,
            outputs + (op.id -> op
              .providesColumns(outputs(op.input1.id), outputs(op.input2.id)))
          )
        case op: Op1 =>
          assert(!outputs.contains(op.id))

          loop(
            ops.tail,
            outputs + (op.id -> op.providesColumns(outputs(op.input.id)))
          )

        case x => throw new RuntimeException("Unexpected op " + x)
      }
    }

    loop(topoSorted, Map.empty)

  }

  private def topologicalSort(root: Op): Seq[Op] = {
    var order = List.empty[Op]
    var marks = Set.empty[UUID]
    var currentParents = Set.empty[UUID]

    def visit(n: Op): Unit =
      if (marks.contains(n.id)) ()
      else {
        if (currentParents.contains(n.id)) {
          println(s"error: loop to ${n.id}")
          ()
        } else {
          currentParents = currentParents + n.id
          val children = n.inputs.map(_.op)
          children.foreach(visit)
          currentParents = currentParents - n.id
          marks = marks + n.id
          order = n :: order
        }
      }

    visit(root)

    order

  }

  sealed trait Mutation {
    def makeChildren(parent: Result): Seq[Result]
  }

  object PushDownFilters extends Mutation {

    def swap(root: Result, filter: Filter, grandChild: Op): Result = {
      val a = filter
      val b = filter.input
      val c = grandChild
      // before c - b - a
      // after c - a - b
      // b can have other children
      val acopy = a.copy(input = grandChild)
      val bcopy = b.replace(c.id, acopy)
      root.replace(a.id, bcopy).asInstanceOf[Result]
    }

    def trySwap(
        root: Result,
        filter: Filter,
        grandChild: Op,
        provided: Map[UUID, Seq[TableColumnRef]]
    ): Option[Result] = {

      val grandChildSatisfiesDependencies =
        (filter.neededColumns.toSet &~ provided(grandChild.id).toSet).isEmpty

      val childTypeOk = filter.input match {
        case _: Filter | _: Projection | _: Product => true
        case x: EquiJoin if x.joinType == InnerJoin => true
        case _                                      => false
      }

      if (childTypeOk && grandChildSatisfiesDependencies)
        Some(swap(root, filter, grandChild))
      else None
    }

    def tryPushFilter(
        root: Result,
        filter: Filter,
        provided: Map[UUID, Seq[TableColumnRef]]
    ): Seq[Result] = {
      val inputInputs = filter.input.inputs
      inputInputs.flatMap { grandChild =>
        trySwap(
          root,
          filter,
          grandChild.op,
          provided
        ).toList
      }
    }

    def makeChildren(parent: Result): Seq[Result] = {
      val sorted = topologicalSort(parent).reverse
      val provided = providedReferences(sorted, parent.boundTables)
      val eligibleFilters = sorted collect { case f: Filter =>
        f
      }

      eligibleFilters.flatMap { filter =>
        tryPushFilter(parent, filter, provided)
      }

    }
  }
  object PushDownInnerJoin extends Mutation {

    def swap(
        root: Result,
        join: EquiJoin,
        child: Op,
        grandChild: Op
    ): Result = {
      val a = join
      val b = child
      val c = grandChild
      // before c - b - a
      // after c - a - b
      // b can have other children
      val acopy =
        if (a.input1.id == child.id) a.copy(input1 = grandChild)
        else a.copy(input2 = grandChild)
      val bcopy = b.replace(c.id, acopy)
      root.replace(a.id, bcopy).asInstanceOf[Result]
    }

    def trySwap(
        root: Result,
        op: EquiJoin,
        child: Op,
        neededColumns: Seq[TableColumnRef],
        grandChild: Op,
        provided: Map[UUID, Seq[TableColumnRef]]
    ): Option[Result] = {

      val grandChildSatisfiesDependencies =
        (neededColumns.toSet &~ provided(grandChild.id).toSet).isEmpty

      val childTypeOk = child match {
        case _: Filter | _: Projection | _: Product => true
        case x: EquiJoin if x.joinType == InnerJoin => true
        case _                                      => false
      }

      if (childTypeOk && grandChildSatisfiesDependencies)
        Some(swap(root, op, child, grandChild))
      else None
    }

    def tryPush(
        root: Result,
        op: EquiJoin,
        provided: Map[UUID, Seq[TableColumnRef]]
    ): Seq[Result] = {
      op.input1.inputs.flatMap { grandChild =>
        trySwap(
          root,
          op,
          op.input1,
          op.neededColumns1,
          grandChild.op,
          provided
        ).toList
      } ++
        op.input2.inputs.flatMap { grandChild =>
          trySwap(
            root,
            op,
            op.input2,
            op.neededColumns2,
            grandChild.op,
            provided
          ).toList
        }
    }

    def makeChildren(root: Result): Seq[Result] = {
      val sorted = topologicalSort(root).reverse
      val provided = providedReferences(sorted, root.boundTables)
      val eligible = sorted collect {
        case f: EquiJoin if f.joinType == InnerJoin =>
          f
      }

      eligible.flatMap { op =>
        tryPush(root, op, provided)
      }

    }
  }

  def depthFirstSearch(start: Result, mutations: List[Mutation], maxDepth:Int) = {

    def children(parent: Result): Seq[Result] =
      mutations.flatMap(_.makeChildren(parent)).distinct

    def loop(
        depth: Int,
        queue: Queue[Result],
        visited: Vector[Result]
    ): Vector[Result] =
      queue.dequeueOption match {
        case Some((head, tail)) if depth < maxDepth =>
          val ch = children(head)
          val nextVisited =
            if (visited.contains(head)) visited else visited :+ head
          loop(depth + 1, tail ++ ch, nextVisited)
        case _ => visited
      }

    loop(0, Queue(start), Vector.empty)

  }
  case class TableEstimate(
      rows: Long,
      columns: Long
  )

  def estimate(root: Result) = {
    val sorted = topologicalSort(root).reverse

    val tables = root.boundTables.toMap

    def makeTableEstimate(tableRef: TableRef): TableEstimate = {
      val table = tables(tableRef)
     
      TableEstimate(table.numRows, table.numCols) 
    }

    def loop(
        ops: Seq[Op],
        outputs: Map[UUID, TableEstimate]
    ): Map[UUID, TableEstimate] = {
      ops.head match {

        case op @ TableOp(tableRef) =>
          assert(!outputs.contains(op.id))
          loop(
            ops.tail,
            outputs + (op.id -> makeTableEstimate(tableRef))
          )

        case op @ Result(input, _) =>
          assert(!outputs.contains(op.id))
          outputs + (op.id -> outputs(input.id))
        case op: Op2 =>
          assert(!outputs.contains(op.id))

          loop(
            ops.tail,
            outputs + (op.id -> op
              .estimate(outputs(op.input1.id), outputs(op.input2.id)))
          )
        case op: Op1 =>
          assert(!outputs.contains(op.id))

          loop(
            ops.tail,
            outputs + (op.id -> op.estimate(outputs(op.input.id)))
          )

        case x => throw new RuntimeException("Unexpected op " + x)
      }
    }

    val outputs = loop(sorted, Map.empty)
    sorted
      .map(v => outputs(v.id))
      .map(estimate => estimate.columns.toDouble * estimate.rows.toDouble)
      .sum

  }

}
