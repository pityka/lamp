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

  case class VariableRef(name: String) {
    override def toString = s"?$name?"
  }
  sealed trait VariableValue
  case class DoubleVariableValue(v: Double) extends VariableValue

  private def extractColumnRefs(table: Table): IndexedSeq[ColumnRef] =
    (0 until table.columns.size map (i =>
      IdxColumnRef(i)
    )) ++ table.nameToIndex.keySet.toSeq.map(s => StringColumnRef(s))

  def interpretBooleanExpression[S: Sc](
      factor: BooleanFactor,
      table: TableWithColumnMapping,
      boundVariables: Map[VariableRef, VariableValue]
  ): STen = factor match {
    case BooleanAtomTrue =>
        STen.ones(List(1),table.table.columns.head.values.options)
    case BooleanAtomFalse =>
        STen.zeros(List(1),table.table.columns.head.values.options)
    case ColumnFunction(columnRefs,_, impl) =>
      val columns = columnRefs
        .map(c =>
          (
            c: TableColumnRef,
            table.table.columns(table.getMatchingColumnIdx(c))
          )
        )
        .toMap
      impl(PredicateHelper(columns, boundVariables))(implicitly[Scope]).values
    case BooleanNegation(factor) =>
      val t = interpretBooleanExpression(factor, table, boundVariables)
      t.logicalNot
    case BooleanExpression(terms) =>
      terms
        .map { term =>
          term.factors
            .map { factor =>
              interpretBooleanExpression(factor, table, boundVariables)
            }
            .toList
            .reduce((a, b) => a.logicalOr(b))
        }
        .toList
        .reduce((a, b) => a.logicalAnd(b))
  }
  def verifyBooleanExpression(
      factor: BooleanFactor,
      table: ColumnSet,
      boundVariables: Map[VariableRef,VariableValue]
  ): Boolean = factor match {
    case BooleanAtomFalse | BooleanAtomTrue => true
    case ColumnFunction(columnRefs, variableRefs,_) =>
      columnRefs
        .forall(c => table.hasMatchingColumn(c)) && 
        variableRefs.forall(c => boundVariables.contains(c))
    case BooleanNegation(factor) =>
      verifyBooleanExpression(factor, table,boundVariables)
    case BooleanExpression(terms) =>
      terms
        .map { term =>
          term.factors
            .map { factor =>
              verifyBooleanExpression(factor, table,boundVariables)
            }
            .toList
            .reduce((a, b) => a && b)
        }
        .toList
        .reduce((a, b) => a && b)
  }
  def columnsReferencedByBooleanExpression(
      factor: BooleanFactor
  ): Seq[TableColumnRef] = factor match {
    case BooleanAtomFalse | BooleanAtomTrue => Nil
    case ColumnFunction(columnRefs,_, _) =>
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
  def variablesReferencedByBooleanExpression(
      factor: BooleanFactor
  ): Seq[VariableRef] = factor match {
    case BooleanAtomFalse | BooleanAtomTrue => Nil
    case ColumnFunction(_,refs, _) =>
      refs.distinct
    case BooleanNegation(factor) =>
      variablesReferencedByBooleanExpression(factor)
    case BooleanExpression(terms) =>
      terms.toList
        .flatMap { term =>
          term.factors.toList.flatMap { factor =>
            variablesReferencedByBooleanExpression(factor)
          }.toList

        }

  }

  trait MatchableTableColumnRefSet {
    def providedColumns: Map[QualifiedTableColumnRef, Int]
    def getMatchingQualifiedColumnRef(
        ref: TableColumnRef
    ): (QualifiedTableColumnRef, Int) = ref match {
      case qr: QualifiedTableColumnRef if providedColumns.contains(qr) =>
        qr -> providedColumns(qr)
      case WildcardColumnRef(column) =>
        val candidates = providedColumns.toSeq.filter(_._1.column == column)
        require(
          candidates.map(_._2).distinct.size == 1,
          s"Multiple columns match unqualified query ${candidates.mkString(", ")}"
        )
        candidates.head
      case _ => throw new RuntimeException(s"Not found $ref")
    }
    def hasMatchingColumn(ref: TableColumnRef): Boolean = ref match {
      case qr: QualifiedTableColumnRef => providedColumns.contains(qr)
      case WildcardColumnRef(column) =>
        val candidates = providedColumns.toSeq.filter(_._1.column == column)
        candidates.map(_._2).distinct.size == 1
    }
  }

  case class ColumnSet(providedColumns: Map[QualifiedTableColumnRef, Int])
      extends MatchableTableColumnRefSet

  case class TableWithColumnMapping(
      table: Table,
      private val columnMap: Map[QualifiedTableColumnRef, Int]
  ) extends MatchableTableColumnRefSet {
    def shiftColumnsUp(i: Int) =
      copy(columnMap = columnMap.map(v => (v._1, v._2 + i)))
    def mapColumns(
        f: ((QualifiedTableColumnRef, Int)) => (QualifiedTableColumnRef, Int)
    ) = copy(columnMap = columnMap.map(f))
    def mergeMaps(other: TableWithColumnMapping) = columnMap ++ other.columnMap
    def providedColumns = columnMap
    def getColumnMapping = columnMap.toSeq
    def columnOrder: Seq[QualifiedTableColumnRef] =
      columnMap.toSeq.sortBy(_._2).map(_._1)

    def getMatchingColumnIdx(ref: TableColumnRef): Int = {
      ref match {
        case qr: QualifiedTableColumnRef => columnMap(qr)
        case WildcardColumnRef(column) =>
          val candidates = columnMap.filter(_._1.column == column)
          require(
            candidates.map(_._2).toSeq.distinct.size == 1,
            s"Multiple columns match unqualified query: ${candidates.mkString(", ")}"
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

        case op @ Result(input, _, _) =>
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
            outputs(op.input2.id)._1,
            root.boundVariables
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
          val result =
            op.impl(outputs(op.input.id)._1, root.boundVariables)(oldScope)
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
  def analyzeReferences(
      topoSorted: Seq[Op],
      tables: Map[TableRef, Table]
  ): Either[String, Map[UUID, ColumnSet]] = {

    def extractColumnRefsFromTableRef(
        tableRef: TableRef,
        boundTable: Option[Table]
    ): ColumnSet = {
      val table = {
      val found = tables.get(tableRef).orElse(boundTable)
      if (found.isEmpty) {
        throw new RuntimeException(s"$tableRef is not bound")
      }
      found.get
      }
      val columnRefs =
        extractColumnRefs(table)
      val mapping = columnRefs.map { ref =>
        ref match {
          case IdxColumnRef(idx) =>
            (QualifiedTableColumnRef(tableRef, ref), idx)
          case StringColumnRef(string) =>
            (QualifiedTableColumnRef(tableRef, ref), table.nameToIndex(string))
        }
      }.toMap
      ColumnSet(mapping)
    }

    def loop(
        ops: Seq[Op],
        outputs: Map[UUID, ColumnSet]
    ): Either[String, Map[UUID, ColumnSet]] = {
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

        case op @ Result(_, _, _) =>
          assert(!outputs.contains(op.id))
          Right(outputs)
        case op: Op2 =>
          assert(!outputs.contains(op.id))
          op
            .analyze(
              outputs(op.input1.id),
              outputs(op.input2.id),
              topoSorted.last.asInstanceOf[Result].boundVariables
            ) match {
            case Right(x) =>
              loop(
                ops.tail,
                outputs + (op.id -> x)
              )
            case Left(error) => Left(error)
          }

        case op: Op1 =>
          assert(!outputs.contains(op.id))
          op.analyze(
            outputs(op.input.id),
            topoSorted.last.asInstanceOf[Result].boundVariables
          ) match {
            case Right(x) =>
              loop(
                ops.tail,
                outputs + (op.id -> x)
              )
            case Left(error) => Left(error)
          }

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
      mutations.flatMap(_.makeChildren(parent).toOption.toList.flatten).distinct

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
          val nextQueue = tail ++ ch.filterNot(ch =>
            tail.contains(ch) || visited.contains(ch)
          )
          loop(depth + 1, nextQueue, nextVisited)
        case _ =>
          visited
      }
    loop(0, Queue(start), Vector.empty)

  }
  case class TableEstimate(
      rows: Long,
      columns: Long
  )

  def estimate(root: Result): Double = {
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

        case op @ Result(input, _, _) =>
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
