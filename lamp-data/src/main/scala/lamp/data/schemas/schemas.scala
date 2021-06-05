package lamp.data.schemas
import com.github.plokhotnyuk.jsoniter_scala.macros._
import com.github.plokhotnyuk.jsoniter_scala.core._

case class TensorDescriptor(
    dims: Seq[Long],
    dataType: Byte,
    location: String,
    byteOffset: Long,
    byteLength: Long
)

object TensorDescriptor {
  implicit val codec: JsonValueCodec[TensorDescriptor] = JsonCodecMaker.make
}

case class TensorList(
    tensors: Seq[TensorDescriptor]
)

object TensorList {
  implicit val codec: JsonValueCodec[TensorList] = JsonCodecMaker.make
}

sealed trait LoopState

case class SimpleLoopState(
    model: TensorList,
    optimizer: TensorList,
    epoch: Int,
    lastValidationLoss: Option[Double],
    minValidationLoss: Option[Double],
    minValidationLossModel: Option[(Int, TensorList)],
    learningCurve: List[(Int, Double, Option[Double])]
) extends LoopState
case class SWALoopState(
    model: TensorList,
    optimizer: TensorList,
    epoch: Int,
    lastValidationLoss: Option[Double],
    minValidationLoss: Option[Double],
    numberOfAveragedModels: Int,
    averagedModels: Option[TensorList],
    learningCurve: List[(Int, Double, Option[Double])]
) extends LoopState
case class SimpleThenSWALoopState(
    simple: Option[SimpleLoopState],
    swa: Option[SWALoopState]
) extends LoopState

object LoopState {
  implicit val codec: JsonValueCodec[LoopState] = JsonCodecMaker.make
}
