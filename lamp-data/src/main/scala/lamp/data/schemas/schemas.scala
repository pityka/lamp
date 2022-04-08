package lamp.data.schemas
import com.github.plokhotnyuk.jsoniter_scala.macros._
import com.github.plokhotnyuk.jsoniter_scala.core._

case class TensorDescriptor(
    dims: Seq[Long],
    dataType: Byte,
    byteOffset: Long,
    byteLength: Long
)

object TensorDescriptor {
  implicit val codec: JsonValueCodec[TensorDescriptor] = JsonCodecMaker.make
}

case class TensorList(
    tensors: Seq[TensorDescriptor],
    location: String,
    byteOffset: Long,
    byteLength: Long
) {
  assert(
    tensors.forall(t => t.byteOffset + t.byteLength <= byteLength),
    s"Some tensor offset+length is out of bound ${tensors
      .map(v => (v.byteOffset, v.byteLength))} total: $byteLength"
  )
}

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
    learningCurve: List[(Int, Double, Option[Double],Option[Double])]
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
    simple: SimpleLoopState,
    swa: Option[SWALoopState]
) extends LoopState

object LoopState {
  implicit val codec: JsonValueCodec[LoopState] = JsonCodecMaker.make
}
