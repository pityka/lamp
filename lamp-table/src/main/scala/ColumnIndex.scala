package lamp.table


import org.saddle.scalar.ScalarTagLong
import org.saddle.scalar.ScalarTagFloat
import org.saddle.scalar.ScalarTagDouble
import org.saddle._
import org.saddle.order._
import org.saddle.Index


sealed trait ColumnIndex[T] {
  def index: Index[T]
  implicit def st: ST[T]
  implicit def ord: ORD[T]
  def uniqueLocations: Array[Vec[Int]] = {
    val u = index.uniques
    u.toVec.toArray.map { l =>
      index.get(l).toVec
    }
  }

}
case class LongIndex(index: Index[Long]) extends ColumnIndex[Long] {
  val st = ScalarTagLong
  val ord = implicitly[ORD[Long]]
}
case class FloatIndex(index: Index[Float]) extends ColumnIndex[Float] {
  val st = ScalarTagFloat
  val ord = implicitly[ORD[Float]]
}
case class DoubleIndex(index: Index[Double]) extends ColumnIndex[Double] {
  val st = ScalarTagDouble
  val ord = implicitly[ORD[Double]]
}
case class StringIndex(index: Index[String]) extends ColumnIndex[String] {
  val st = implicitly[ST[String]]
  val ord = implicitly[ORD[String]]
}
