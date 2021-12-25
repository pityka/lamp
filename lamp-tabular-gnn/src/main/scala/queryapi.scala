package lamp.tgnn
import RelationalAlgebra._
import lamp._
import org.saddle.index.InnerJoin
import org.saddle.index.JoinType
import scala.language.dynamics
import scala.collection.immutable
trait StackOps {
  def project(projectTo: ColumnFunctionWithOutputRef*) =
    StackOp1Token(in => Projection(in, projectTo))
  def filter(expr: BooleanFactor) = StackOp1Token(in => Filter(in, expr))
  def product = StackOp2Token((in1, in2) => Product(in1, in2))
  def union = StackOp2Token((in1, in2) => Union(in1, in2))
  def innerEquiJoin(
      thisColumn: TableColumnRef,
      thatColumn: TableColumnRef
  ) = StackOp2Token((in1, in2) =>
    EquiJoin(in1, in2, thisColumn, thatColumn, InnerJoin)
  )
  def aggregate(groupBy: TableColumnRef*)(
      aggregates: ColumnFunctionWithOutputRef*
  ) = StackOp1Token(in => Aggregate(in, groupBy, aggregates))
  def pivot(rowKeys: TableColumnRef, colKeys: TableColumnRef)(
      aggregate: ColumnFunction
  ) = StackOp1Token(in => Pivot(in, rowKeys, colKeys, aggregate))
}

sealed trait StackToken
case class OpToken(op: Op) extends StackToken
case class StackOp1Token(stackOp1: Op => Op) extends StackToken
case class StackOp2Token(stackOp2: (Op, Op) => Op) extends StackToken
case class ListToken(list: TokenList) extends StackToken

case class TokenList(head: Op, list: List[StackToken]) {
  def ~(token: StackToken) = TokenList(head, list.appended(token))
  def ~(token: Table) =
    TokenList(head, list.appended(OpToken(syntax.TableSyntax(token).query)))
  def ~(token: TableRef) = TokenList(head, list.appended(OpToken(token.asOp)))
  def doneAndInterpret(tables: (TableRef, Table)*)(implicit scope: Scope) =
    compile.doneAndInterpret(tables: _*)
  def compile: Result = {

    def loop(stack: Stack, remaining: List[StackToken]): Result =
      remaining match {
        case OpToken(head) :: next =>
          loop(stack.push(head), next)
        case StackOp1Token(f) :: next =>
          val (operand, poppedStack) = stack.pop
          val op2 = f(operand)
          loop(poppedStack.push(op2), next)
        case StackOp2Token(f) :: next =>
          println(stack)
          val (operand2, poppedStack1) = stack.pop
          val (operand1, poppedStack2) = poppedStack1.pop
          val op2 = f(operand1, operand2)
          loop(poppedStack2.push(op2), next)
        case ListToken(TokenList(head, tail)) :: next =>
          loop(stack, OpToken(head) :: tail ::: next)
        case immutable.Nil => stack.pop._1.done
      }

    println(this)
    loop(Stack(Vector(head)), list)

  }
}

case class Stack(stack: Vector[Op]) {

  def push(op: Op) = copy(stack = stack.appended(op))
  def pop = (stack.last, copy(stack = stack.dropRight(1)))

}

object syntax {

  implicit class TableSyntax(table: Table) {
    def ref: TableRef = BoundTableRef(table)
    def query: TableOp = {
      TableOp(table.ref, Some(table))
    }
    def ~(that: Op) = TokenList(table.query, List(OpToken(that)))
    def ~(that: Table) = TokenList(table.query, List(OpToken(that.query)))
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

object Q extends Dynamic with StackOps {
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
