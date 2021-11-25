package lamp.tgnn
import lamp.tgnn.RelationalAlgebra._

trait TableColumnRefSyntax { self: TableColumnRef =>
  def select = Q(this) { input => _ =>
    input(this)
  }
  def self = select.asSelf

  def ===(other: TableColumnRef) = Q(this, other) { input => implicit scope =>
    Table.Column.bool(input(this).values.equ(input(other).values).castToLong)
  }
  def !=(other: TableColumnRef) = Q(this, other) { input => implicit scope =>
    Table.Column.bool(
      input(this).values.equ(input(other).values).logicalNot.castToLong
    )
  }
  def <(other: TableColumnRef) = Q(this, other) { input => implicit scope =>
    Table.Column.bool(input(this).values.lt(input(other).values).castToLong)
  }
  def >(other: TableColumnRef) = Q(this, other) { input => implicit scope =>
    Table.Column.bool(input(this).values.gt(input(other).values).castToLong)
  }
  def <=(other: TableColumnRef) = Q(this, other) { input => implicit scope =>
    Table.Column.bool(input(this).values.le(input(other).values).castToLong)
  }
  def >=(other: TableColumnRef) = Q(this, other) { input => implicit scope =>
    Table.Column.bool(input(this).values.ge(input(other).values).castToLong)
  }

  def ===(other: Double) = Q(this) { input => implicit scope =>
    Table.Column.bool(input(this).values.equ(other).castToLong)
  }
  def !=(other: Double) = Q(this) { input => implicit scope =>
    Table.Column.bool(input(this).values.equ(other).logicalNot.castToLong)
  }
  def <(other: Double) = Q(this) { input => implicit scope =>
    Table.Column.bool(input(this).values.lt(other).castToLong)
  }
  def >(other: Double) = Q(this) { input => implicit scope =>
    Table.Column.bool(input(this).values.gt(other).castToLong)
  }
  def <=(other: Double) = Q(this) { input => implicit scope =>
    Table.Column.bool(input(this).values.le(other).castToLong)
  }
  def >=(other: Double) = Q(this) { input => implicit scope =>
    Table.Column.bool(input(this).values.ge(other).castToLong)
  }

}
