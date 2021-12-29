package lamp.tgnn
import lamp.tgnn.RelationalAlgebra._

trait TableColumnRefSyntax { self: TableColumnRef =>
  def select = Q("select")(this) { input => _ =>
    input(this)
  }
  def identity = select

  def isMissing = Q("isna")(this) { input => implicit scope =>
    Table.Column.bool(input(this).missingnessMask.castToLong)
  }
  def isNotMissing = Q("isnonna")(this) { input => implicit scope =>
    Table.Column.bool(input(this).missingnessMask.logicalNot.castToLong)
  }
  def ===(other: TableColumnRef) = Q("eq")(this, other) { input => implicit scope =>
    Table.Column.bool(input(this).values.equ(input(other).values).castToLong)
  }
  def !=(other: TableColumnRef) = Q("neq")(this, other) { input => implicit scope =>
    Table.Column.bool(
      input(this).values.equ(input(other).values).logicalNot.castToLong
    )
  }
  def <(other: TableColumnRef) = Q("lt")(this, other) { input => implicit scope =>
    Table.Column.bool(input(this).values.lt(input(other).values).castToLong)
  }
  def >(other: TableColumnRef) = Q("gt")(this, other) { input => implicit scope =>
    Table.Column.bool(input(this).values.gt(input(other).values).castToLong)
  }
  def <=(other: TableColumnRef) = Q("le")(this, other) { input => implicit scope =>
    Table.Column.bool(input(this).values.le(input(other).values).castToLong)
  }
  def >=(other: TableColumnRef) = Q("ge")(this, other) { input => implicit scope =>
    Table.Column.bool(input(this).values.ge(input(other).values).castToLong)
  }

  def ===(other: Double) = Q("eq")(this) { input => implicit scope =>
    Table.Column.bool(input(this).values.equ(other).castToLong)
  }
  def !=(other: Double) = Q("neq")(this) { input => implicit scope =>
    Table.Column.bool(input(this).values.equ(other).logicalNot.castToLong)
  }
  def <(other: Double) = Q("le")(this) { input => implicit scope =>
    Table.Column.bool(input(this).values.lt(other).castToLong)
  }
  def >(other: Double) = Q("gt")(this) { input => implicit scope =>
    Table.Column.bool(input(this).values.gt(other).castToLong)
  }
  def <=(other: Double) = Q("le")(this) { input => implicit scope =>
    Table.Column.bool(input(this).values.le(other).castToLong)
  }
  def >=(other: Double) = Q("ge")(this) { input => implicit scope =>
    Table.Column.bool(input(this).values.ge(other).castToLong)
  }

  def ===(other: VariableRef) = Q.fun("eq")(this)(other) { input => implicit scope =>
    input.variable(other) match {
      case DoubleVariableValue(v) =>
        Table.Column.bool(input(this).values.equ(v).castToLong)
    }
  }
  def !=(other: VariableRef) = Q.fun("neq")(this)(other) { input => implicit scope =>
    input.variable(other) match {
      case DoubleVariableValue(v) =>
        Table.Column.bool(input(this).values.equ(v).logicalNot.castToLong)
    }
  }

  def <(other: VariableRef) = Q.fun("lt")(this)(other) { input => implicit scope =>
    input.variable(other) match {
      case DoubleVariableValue(v) =>
        Table.Column.bool(input(this).values.lt(v).castToLong)
    }
  }
  def >(other: VariableRef) = Q.fun("gt")(this)(other) { input => implicit scope =>
    input.variable(other) match {
      case DoubleVariableValue(v) =>
        Table.Column.bool(input(this).values.gt(v).castToLong)
    }
  }
  def <=(other: VariableRef) = Q.fun("le")(this)(other) { input => implicit scope =>
    input.variable(other) match {
      case DoubleVariableValue(v) =>
        Table.Column.bool(input(this).values.le(v).castToLong)
    }
  }
  def >=(other: VariableRef) = Q.fun("ge")(this)(other) { input => implicit scope =>
    input.variable(other) match {
      case DoubleVariableValue(v) =>
        Table.Column.bool(input(this).values.ge(v).castToLong)
    }
  }

}
