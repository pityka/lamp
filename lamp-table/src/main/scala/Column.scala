package lamp.table

import lamp._
// import org.saddle.scalar.ScalarTagLong

import org.saddle._
import org.saddle.order._
import org.saddle.Index
import lamp.saddle._
import org.saddle.scalar.ScalarTagLong

case class Column(
    values: STen,
    tpe: ColumnDataType,
    index: Option[ColumnIndex[_]]
) extends ColumnOps {

  def table = Table(this)
  def tableWithName(s: String) = Table(this).rename(0, s)

  def missingnessMask[S: Sc]: Column = {
    tpe match {
      case _: BooleanColumnType | _: DateTimeColumnType | I64ColumnType =>
        Column(values.equ(ScalarTagLong.missing))
      case _: TextColumnType =>
        Column(values.equ(ScalarTagLong.missing).sum(1, false).gt(0d))
      case F32ColumnType | F64ColumnType =>
        Column(values.isnan)
    }

  }

  private[table] def indexAs[A] = index.asInstanceOf[Option[ColumnIndex[A]]]

  def withIndex = if (index.isDefined) this else copy(index = Some(makeIndex))
  def indexed = withIndex

  def toVec: Vec[_] = tpe match {
    case _: DateTimeColumnType =>
      values.toLongVec

    case _: BooleanColumnType =>
      values.toLongVec

    case TextColumnType(_, pad, vocabulary) =>
      val reverseVocabulary = vocabulary.map(_.map(_.swap))
      val vec = values.toLongMat.rows.map { row =>
        if (row.count == 0) null
        else
          row
            .filter(_ != pad)
            .map(l => reverseVocabulary.map(_.apply(l)).getOrElse(l.toChar))
            .toArray
            .mkString
      }.toVec
      vec
    case I64ColumnType =>
      val m = Scope.unsafe { implicit scope =>
        if (values.numel == 0) Vec.empty[Long]
        else
          values.view(values.shape(0), -1).select(1, 0).toLongVec
      }
      m
    case F32ColumnType =>
      val m = Scope.unsafe { implicit scope =>
        if (values.numel == 0) Vec.empty[Float]
        else
          values.view(values.shape(0), -1).select(1, 0).toFloatVec
      }
      m
    case F64ColumnType =>
      val m = Scope.unsafe { implicit scope =>
        if (values.numel == 0) Vec.empty[Double]
        else
          values.view(values.shape(0), -1).select(1, 0).toVec
      }
      m
  }
  def makeIndex: ColumnIndex[_] = tpe match {
    case _: DateTimeColumnType =>
      LongIndex(Index(values.toLongVec))

    case _: BooleanColumnType =>
      LongIndex(Index(values.toLongVec))

    case TextColumnType(_, pad, vocabulary) =>
      val reverseVocabulary = vocabulary.map(_.map(_.swap))
      val vec = values.toLongMat.rows.map { row =>
        if (row.count == 0) null
        else
          row
            .filter(_ != pad)
            .map(l => reverseVocabulary.map(_.apply(l)).getOrElse(l.toChar))
            .toArray
            .mkString
      }.toVec
      StringIndex(Index(vec))
    case I64ColumnType =>
      val m = Scope.unsafe { implicit scope =>
        if (values.numel == 0L) Vec.empty[Long]
        else values.view(values.shape(0), -1).select(1, 0).toLongVec
      }
      LongIndex(Index(m))
    case F32ColumnType =>
      val m = Scope.unsafe { implicit scope =>
        if (values.numel == 0L) Vec.empty[Float]
        else
          values.view(values.shape(0), -1).select(1, 0).toFloatVec
      }
      FloatIndex(Index(m))
    case F64ColumnType =>
      val m = Scope.unsafe { implicit scope =>
        if (values.numel == 0L) Vec.empty[Double]
        else
          values.view(values.shape(0), -1).select(1, 0).toVec
      }
      DoubleIndex(Index(m))
  }

  // def whereEquals[A](other: Column)(implicit scope: Scope): STen = {
  //   if (index.isDefined && other.index.isDefined) {
  //     val i1 = indexAs[A].get.index
  //     val i2 = other.indexAs[A].get.index
  //     if (i1.isUnique && i2.isUnique) {
  //       val reindexer = i1.intersect(i2)
  //       STen.fromLongArray(reindexer.lTake.get.map(_.toLong))
  //     } else values.equ(other.values).where.head
  //   } else
  //     values.equ(other.values).where.head

  // }

  def equijoin[IndexType](
      other: Column,
      how: org.saddle.index.JoinType = org.saddle.index.InnerJoin
  )(implicit scope: Scope): (Column, Column) = {
    val t = this.table.equijoin(0, other.table, 0, how)
    (t.colAt(0), t.colAt(1))
  }

  def distinct(implicit scope: Scope) = this.table.distinct.colAt(0)
  def select[S: Sc](idx: STen): Column =
    Column(values.indexSelect(dim = 0, index = idx), tpe, None)

  def product(other: Column)(implicit scope: Scope): (Column, Column) = {
    val t = this.table.product(other.table)
    (t.colAt(0), t.colAt(1))
  }

  def filter(predicate: Column)(implicit
      scope: Scope
  ): Column =
    Scope { implicit scope =>
      val indices = predicate.values.where.head
      this.select(indices)
    }

}
object Column {
  implicit val movable: Movable[Column] = Movable.by(_.values)
  def bool(s: STen) = Column(s, BooleanColumnType(), None)

  def apply(s: STen): Column =
    Column(s, Column.dataTypeFromScalarTypeByte(s.scalarTypeByte), None)

  private[table] def dataTypeFromScalarTypeByte(s: Byte) = s match {
    case 11 => BooleanColumnType()
    case 7  => F64ColumnType
    case 6  => F32ColumnType
    case 4  => I64ColumnType
  }
}
