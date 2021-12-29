package lamp.tgnn
import RelationalAlgebra._
import cats.data.NonEmptyList
import lamp._
sealed trait BooleanFactor {
  def negate: BooleanFactor = BooleanNegation(this)
  def or(that: BooleanFactor*): BooleanFactor = BooleanExpression(
    NonEmptyList(BooleanTerm(NonEmptyList(this, that.toList)), Nil)
  )
  def ===(that: BooleanFactor): BooleanFactor = BooleanExpression(
    NonEmptyList(
      BooleanTerm(
        NonEmptyList((this.and(that)).or(this.negate.and(that.negate)), Nil)
      ),
      Nil
    )
  )
  def and(that: BooleanFactor*): BooleanFactor = BooleanExpression(
    NonEmptyList(
      BooleanTerm(NonEmptyList(this, Nil)),
      that.toList.map(f => BooleanTerm(NonEmptyList(f, Nil)))
    )
  )
}

case class ColumnFunction(
    columnRefs: Seq[TableColumnRef],
    variableRefs: Seq[VariableRef],
    impl: PredicateHelper => Scope => Table.Column
) extends BooleanFactor {
  def as(ref: QualifiedTableColumnRef) = ColumnFunctionWithOutputRef(this, ref)
  override def toString =
    s"[${columnRefs.mkString(",")} ${variableRefs.mkString(", ")}]"
}

case object BooleanAtomTrue extends BooleanFactor {
  override def toString = "TRUE"
}
case object BooleanAtomFalse extends BooleanFactor {
  override def toString = "FALSE"
}

case class BooleanNegation(factor: BooleanFactor) extends BooleanFactor {
  override def toString = factor match {
    case BooleanAtomTrue  => "FALSE"
    case BooleanAtomFalse => "TRUE"
    case _                => s"\u00AC$factor"
  }
}
case class BooleanTerm(factors: NonEmptyList[BooleanFactor]) { // or
  override def toString = if (factors.size == 1) factors.head.toString
  else factors.toList.mkString("(", " \u2228 ", ")")
}
case class BooleanExpression(terms: NonEmptyList[BooleanTerm]) // and
    extends BooleanFactor {
  override def toString = if (terms.size == 1) terms.head.toString
  else terms.toList.mkString("(", " \u2227 ", ")")
}
