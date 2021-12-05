package lamp.tgnn

import java.util.UUID
import lamp._
import scala.collection.immutable.Queue
import scala.language.dynamics

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
  trait TableRef extends Dynamic {

    def selectDynamic(field: String) = col(field)
    def selectDynamic(field: Int) = col(field)

    def col(i: Int): QualifiedTableColumnRef =
      QualifiedTableColumnRef(this, IdxColumnRef(i))
    def col(s: String): QualifiedTableColumnRef =
      QualifiedTableColumnRef(this, StringColumnRef(s))

    def asOp = TableOp(this, None)
    def scan = asOp
  }
  case class AliasTableRef(alias: String) extends TableRef {
    override def toString = alias
  }
  case class BoundTableRef(table: Table) extends TableRef {
    override def toString = "T" + table.hashCode().toInt
  }
  sealed trait TableColumnRef extends TableColumnRefSyntax {
    def column: ColumnRef
    override def toString = s"*.$column"

    def matches(other: TableColumnRef): Boolean = (this, other) match {
      case (a: QualifiedTableColumnRef, b: QualifiedTableColumnRef) => a == b
      case _ => this.column == other.column
    }
  }
  case class QualifiedTableColumnRef(table: TableRef, column: ColumnRef)
      extends TableColumnRef {
    override def toString = s"$table.$column"
    def self = select.as(this)
  }
  case class WildcardColumnRef(column: ColumnRef) extends TableColumnRef

  case class ColumnFunctionWithOutputRef(
      function: ColumnFunction,
      outputRef: QualifiedTableColumnRef
  ) {
    override def toString = function.toString
  }

  private def extractColumnRefs(table: Table): IndexedSeq[ColumnRef] =
    (0 until table.columns.size map (i =>
      IdxColumnRef(i)
    )) ++ table.nameToIndex.keySet.toSeq.map(s => StringColumnRef(s))

  def interpretBooleanExpression[S: Sc](
      factor: BooleanFactor,
      table: TableWithColumnMapping
  ): STen = factor match {
    case ColumnFunction(columnRefs, impl) =>
      val columns = columnRefs
        .map(c =>
          (
            c: TableColumnRef,
            table.table.columns(table.getMatchingColumnIdx(c))
          )
        )
        .toMap
      impl(PredicateHelper(columns))(implicitly[Scope]).values
    case BooleanNegation(factor) =>
      val t = interpretBooleanExpression(factor, table)
      t.logicalNot
    case BooleanExpression(terms) =>
      terms
        .map { term =>
          term.factors
            .map { factor =>
              interpretBooleanExpression(factor, table)
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
      private val columnMap: Map[QualifiedTableColumnRef, Int]
  ) {
    def shiftColumnsUp(i: Int) =
      copy(columnMap = columnMap.map(v => (v._1, v._2 + i)))
    def mapColumns(
        f: ((QualifiedTableColumnRef, Int)) => (QualifiedTableColumnRef, Int)
    ) = copy(columnMap = columnMap.map(f))
    def mergeMaps(other: TableWithColumnMapping) = columnMap ++ other.columnMap
    def providedColumns = columnMap.keySet
    def getColumnMapping = columnMap.toSeq
    def columnOrder: Seq[QualifiedTableColumnRef] =
      columnMap.toSeq.sortBy(_._2).map(_._1)
    def getMatchingQualifiedColumnRef(
        ref: TableColumnRef
    ): QualifiedTableColumnRef = ref match {
      case qr: QualifiedTableColumnRef if columnMap.contains(qr) => qr
      case WildcardColumnRef(column) =>
        val candidates = columnMap.filter(_._1.column == column)
        require(
          candidates.size == 1,
          "Multiple columns match unqualified query"
        )
        candidates.head._1
      case _ => throw new RuntimeException(s"Not found $ref")
    }
    def hasMatchingColumn(ref: TableColumnRef): Boolean = ref match {
      case qr: QualifiedTableColumnRef => columnMap.contains(qr)
      case WildcardColumnRef(column) =>
        val candidates = columnMap.filter(_._1.column == column)
        candidates.size == 1
    }
    def getMatchingColumnIdx(ref: TableColumnRef): Int = {
      ref match {
        case qr: QualifiedTableColumnRef => columnMap(qr)
        case WildcardColumnRef(column) =>
          val candidates = columnMap.filter(_._1.column == column)
          require(
            candidates.size == 1,
            "Multiple columns match unqualified query"
          )
          candidates.head._2
      }
    }
  }
  object TableWithColumnMapping {
    implicit val movable: Movable[TableWithColumnMapping] =
      Movable.by(v => v.table)

  }

  def interpret(root: Result)(implicit
      scope: Scope
  ): Table = Scope { scope =>
    val sorted = topologicalSort(root).reverse

    val sortedIds = sorted.map(_.id)
    val parents: Map[UUID, Seq[UUID]] = sorted
      .flatMap(op => op.inputs.map(v => v.op.id -> op.id))
      .groupBy(_._1)
      .map(v => (v._1, v._2.map(_._2)))
    val releaseStepNumber = sorted
      .map { op =>
        (
          op.id,
          parents
            .get(op.id)
            .toList
            .flatten
            .map(v => sortedIds.indexOf(v))
            .maxOption
        )
      }
      .toMap
      .filter(_._2.isDefined)
      .map(v => (v._1, v._2.get))

    def makeMapping(
        tableRef: TableRef,
        boundTable: Option[Table]
    ): TableWithColumnMapping = {
      val tables = root.boundTables.toMap
      val table = tables.get(tableRef).orElse(boundTable).get
      val columnRefs =
        extractColumnRefs(table).map(QualifiedTableColumnRef(tableRef, _))
      val mapping = columnRefs.map { ref =>
        ref.column match {
          case IdxColumnRef(idx) => (ref, idx)
          case StringColumnRef(string) =>
            (ref, table.nameToIndex(string))
        }
      }.toMap
      TableWithColumnMapping(table, mapping)
    }

    def loop(
        step: Int,
        ops: Seq[Op],
        outputs: Map[UUID, (TableWithColumnMapping, Scope)]
    ): Table = {
      val releasable = outputs.keySet.filter(id => releaseStepNumber(id) < step)
      releasable.foreach { id =>
        outputs(id)._2.release()
      }
      ops.head match {

        case op @ TableOp(tableRef, boundTable) =>
          assert(!outputs.contains(op.id))
          loop(
            step + 1,
            ops.tail,
            outputs + (
              (
                op.id,
                (makeMapping(tableRef, boundTable), Scope.free)
              )
            ) -- releasable
          )

        case op @ Result(input, _) =>
          assert(!outputs.contains(op.id))
          val (table, oldScope) = outputs(input.id)
          oldScope.moveInto(scope, table)
          oldScope.release()
          table.table
        case op: Op2 =>
          assert(!outputs.contains(op.id))

          val oldScope1 = outputs(op.input1.id)._2
          val oldScope2 = outputs(op.input2.id)._2

          val result = op.impl(
            outputs(op.input1.id)._1,
            outputs(op.input2.id)._1
          )(oldScope1)
          val newScope = Scope.free
          oldScope1.moveInto(oldScope2, result)
          oldScope2.moveInto(newScope, result)

          loop(
            step + 1,
            ops.tail,
            outputs + ((op.id, (result, newScope))) -- releasable
          )
        case op: Op1 =>
          assert(!outputs.contains(op.id))

          val oldScope = outputs(op.input.id)._2
          val result = op.impl(outputs(op.input.id)._1)(oldScope)
          val newScope = Scope.free
          oldScope.moveInto(newScope, result)

          loop(
            step + 1,
            ops.tail,
            outputs + ((op.id, (result, newScope))) -- releasable
          )

        case x => throw new RuntimeException("Unexpected op " + x)
      }
    }

    loop(0, sorted, Map.empty)

  }
  def providedReferences(topoSorted: Seq[Op], tables: Map[TableRef, Table]) = {

    def extractColumnRefsFromTableRef(
        tableRef: TableRef,
        boundTable: Option[Table]
    ): Seq[TableColumnRef] = {
      val table = tables.get(tableRef).orElse(boundTable).get
      val columnRefs =
        extractColumnRefs(table).map(QualifiedTableColumnRef(tableRef, _))

      columnRefs
    }

    def loop(
        ops: Seq[Op],
        outputs: Map[UUID, Seq[TableColumnRef]]
    ): Map[UUID, Seq[TableColumnRef]] = {
      ops.head match {

        case op @ TableOp(tableRef, boundTable) =>
          assert(!outputs.contains(op.id))
          loop(
            ops.tail,
            outputs + (op.id -> extractColumnRefsFromTableRef(
              tableRef,
              boundTable
            ))
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

  def topologicalSort(root: Op): Seq[Op] = {
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

  def depthFirstSearch(
      start: Result,
      mutations: List[Mutation],
      maxDepth: Int
  ) = {

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

    def makeTableEstimate(
        tableRef: TableRef,
        boundTable: Option[Table]
    ): TableEstimate = {
      val table = tables.get(tableRef).orElse(boundTable).get

      TableEstimate(table.numRows, table.numCols)
    }

    def loop(
        ops: Seq[Op],
        outputs: Map[UUID, TableEstimate]
    ): Map[UUID, TableEstimate] = {
      ops.head match {

        case op @ TableOp(tableRef, boundTable) =>
          assert(!outputs.contains(op.id))
          loop(
            ops.tail,
            outputs + (op.id -> makeTableEstimate(tableRef, boundTable))
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
