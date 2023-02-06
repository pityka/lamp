package lamp 

import scala.language.implicitConversions

case class ColumnSelection(e: Either[String,Int]) 

package object table {
  implicit def i2CS(i:Range) : Seq[ColumnSelection]= i.map(i => ColumnSelection(Right(i)))
  implicit def i2CS(i:Int*) : Seq[ColumnSelection]= i.map(i => ColumnSelection(Right(i)))
  implicit def i2CS(i:Int) : ColumnSelection= ColumnSelection(Right(i))
  implicit def s2CS(i:String) : ColumnSelection= ColumnSelection(Left(i))
}
