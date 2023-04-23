package lamp.table

import lamp._

trait ColumnOps { self: Column =>

  /** Fills the tensor with the given `fill` value in the locations indicated by
    * the `mask` boolean mask.
    */
  def maskFill[S: Sc](mask: Column, fill: Double): Column =
    Column(values.maskFill(mask.values, fill))

  /** Fills the tensor with the given `fill` value in the locations indicated by
    * the `mask` boolean mask.
    */
  def maskFill[S: Sc](mask: Column, fill: Long): Column =
    Column(values.maskFill(mask.values, fill))

  /** Returns a boolean tensors of the same shape, indicating equality with the
    * other tensor.
    */
  def equ[S: Sc](other: Column): Column =
    Column(values.equ(other.values))

  /** Returns a boolean tensors of the same shape, indicating equality with the
    * other value.
    */
  def equ[S: Sc](other: Double) =
    Column(values.equ(other))

  /** Returns a boolean tensors of the same shape, indicating equality with the
    * other value.
    */
  def equ[S: Sc](other: Long) =
    Column(values.equ(other))

  /** Casts to byte. signed 8-bit integer (like Scala's Byte) This is called
    * Char in libtorch
    */
  def castToByte[S: Sc] = Column(values.castToByte)

  /** Casts to float */
  def castToFloat[S: Sc] = Column(values.castToFloat)

  /** Casts to double */
  def castToDouble[S: Sc] = Column(values.castToDouble)

  /** Casts to long */
  def castToLong[S: Sc] = Column(values.castToLong)

  /** Casts to long */
  def castToBool[S: Sc] = Column(values.castToBool)

  /** Adds to tensors. */
  def +[S: Sc](other: Column) =
    this.tpe match {
      case F64ColumnType         => Column(values + other.values)
      case F32ColumnType         => Column(values + other.values)
      case _: DateTimeColumnType => ???
      case _: TextColumnType     => ???
      case I64ColumnType         => Column(values add_l other.values)
      case _: BooleanColumnType  => Column(values.logicalOr(other.values))
    }

  /** Subtracts from tensor. */
  def -[S: Sc](other: Column) =
    this.tpe match {
      case F64ColumnType         => Column(values - other.values)
      case F32ColumnType         => Column(values - other.values)
      case _: DateTimeColumnType => ???
      case _: TextColumnType     => ???
      case I64ColumnType         => Column(values sub_l other.values)
      case _: BooleanColumnType  => ???
    }

  /** Adds a scalar to all elements. */
  def +[S: Sc](other: Double) =
    this.tpe match {
      case F64ColumnType         => Column(values + other)
      case F32ColumnType         => Column(values + other)
      case _: DateTimeColumnType => ???
      case _: TextColumnType     => ???
      case I64ColumnType         => ???
      case _: BooleanColumnType =>
        Column(values.logicalOr(STen.scalarDouble(other, values.options)))
    }

  /** Adds a scalar to all elements. */
  def +[S: Sc](other: Long) =
    this.tpe match {
      case F64ColumnType => Column(values + other)
      case F32ColumnType => Column(values + other)
      case _: DateTimeColumnType =>
        Column(values add_l STen.scalarLong(other, values.options))
      case _: TextColumnType =>
        Column(values add_l STen.scalarLong(other, values.options))
      case I64ColumnType =>
        Column(values add_l STen.scalarLong(other, values.options))
      case _: BooleanColumnType =>
        Column(values.logicalOr(STen.scalarLong(other, values.options)))
    }

  /** Multiplication */
  def *[S: Sc](other: STen) = Column(values * other.values)

  def *[S: Sc](other: Long) = Column(values * other)
  def *[S: Sc](other: Double) = Column(values * other)
  /* Division */
  def /[S: Sc](other: STen) = Column(values / other.values)

  def /[S: Sc](other: Long) = Column(values / other)
  def /[S: Sc](other: Double) = Column(values / other)

  def sign[S: Sc] = Column(values.sign)
  def exp[S: Sc] = Column(values.exp)
  def log[S: Sc] = Column(values.log)
  def log1p[S: Sc] = Column(values.log1p)
  def sin[S: Sc] = Column(values.sin)
  def cos[S: Sc] = Column(values.cos)
  def tan[S: Sc] = Column(values.tan)
  def atan[S: Sc] = Column(values.atan)
  def acos[S: Sc] = Column(values.acos)
  def asin[S: Sc] = Column(values.asin)
  def sqrt[S: Sc] = Column(values.sqrt)
  def square[S: Sc] = Column(values.square)
  def abs[S: Sc] = Column(values.abs)
  def ceil[S: Sc] = Column(values.ceil)
  def floor[S: Sc] = Column(values.floor)
  def reciprocal[S: Sc] = Column(values.reciprocal)
  def remainder[S: Sc](other: Column) = Column(values.remainder(other.values))

  def pow[S: Sc](exponent: Double) = Column(values.pow(exponent))

  /** Returns a long tensors with the argsort of the given dimension.
    *
    * Indexing the given dimension by the returned tensor would result in a
    * sorted order.
    */
  def argsort[S: Sc](descending: Boolean) =
    tpe match {
      case _: TextColumnType => ???
      case _                 => Column(values.argsort(dim = 0, descending=descending, stable=true))
    }

  /** Return a boolean tensor indicating element-wise greater-than. */
  def >[S: Sc](other: Column) =
    Column(values gt other.values)
  def <[S: Sc](other: Column) =
    Column(values lt other.values)
  def <=[S: Sc](other: Column) =
    Column(values le other.values)
  def >=[S: Sc](other: Column) =
    Column(values ge other.values)
  def <>[S: Sc](other: Column) =
    Column(values ne other.values)

  /** Return a boolean tensor with element-wise logical and. */
  def and[S: Sc](other: Column) =
    Column(values logicalAnd other.values)

  def or[S: Sc](other: Column) =
    Column(values logicalOr other.values)
  def xor[S: Sc](other: Column) =
    Column(values logicalXor other.values)
  def not[S: Sc] =
    Column(values.logicalNot)
  def isNaN[S: Sc] =
    Column(values.isnan)
  def nanToNum[S: Sc] =
    Column(values.nanToNum)

  def round[S: Sc] = Column(values.round)

}
