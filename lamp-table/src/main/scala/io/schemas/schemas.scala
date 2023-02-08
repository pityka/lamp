package lamp.table.io.schemas
import com.github.plokhotnyuk.jsoniter_scala.macros._
import com.github.plokhotnyuk.jsoniter_scala.core._
import lamp.data.schemas.TensorList

sealed trait ColumnDataTypeDescriptor
case class TextColumnType(
    maxLength: Int,
    pad: Long,
    vocabulary: Option[Map[Char, Long]]
) extends ColumnDataTypeDescriptor
case object DateTimeColumnType extends ColumnDataTypeDescriptor
case object BooleanColumnType extends ColumnDataTypeDescriptor
case object I64ColumnType extends ColumnDataTypeDescriptor
case object F64ColumnType extends ColumnDataTypeDescriptor
case object F32ColumnType extends ColumnDataTypeDescriptor

case class TableDescriptor(
    columnTypes: List[ColumnDataTypeDescriptor],
    columnNames: List[String],
    columnValues: TensorList
)

object TableDescriptor {
  implicit val codec: JsonValueCodec[TableDescriptor] = JsonCodecMaker.make
}
