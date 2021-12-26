package lamp.tgnn
import RelationalAlgebra._
import cats.data.NonEmptyList
import lamp._
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

case class PredicateHelper(
    map: Map[TableColumnRef, Table.Column],
    variables: Map[VariableRef, VariableValue]
) {
  def apply(t: TableColumnRef) = map(t)
  def variable(ref: VariableRef) = variables(ref)
}

case class ColumnFunction(
    columnRefs: Seq[TableColumnRef],
    variableRefs: Seq[VariableRef],
    impl: PredicateHelper => Scope => Table.Column
) extends BooleanFactor {
  def as(ref: QualifiedTableColumnRef) = ColumnFunctionWithOutputRef(this, ref)
  override def toString = s"[${columnRefs.mkString(",")} ${variableRefs.mkString(", ")}]"
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
