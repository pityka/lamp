package lamp.data.schemas
import com.github.plokhotnyuk.jsoniter_scala.macros._
import com.github.plokhotnyuk.jsoniter_scala.core._
import lamp.nn

case class ByteSegmentEncoding(
    encoding: Vector[(Vector[Byte], Int)],
    unknownToken: Int,
    unknownByte: Byte
)

object ByteSegmentEncoding {
  implicit val codec: JsonValueCodec[ByteSegmentEncoding] = JsonCodecMaker.make
}

private[lamp] object Schemas {

  /** dataType is pytorch scalartype:
    *
    *   - 1 I8
    *   - 2 I16
    *   - 3 I32
    *   - 4 I64
    *   - 5 FP16
    *   - 6 FP32
    *   - 7 FP64
    *   - 11 bool
    *   - 16 BF16
    */
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

  sealed trait LoopState {
    def locations: Seq[String]
  }

  case class SimpleLoopState(
      model: TensorList,
      optimizer: TensorList,
      epoch: Int,
      lastValidationLoss: Option[Double],
      minValidationLoss: Option[Double],
      minValidationLossModel: Option[(Int, TensorList)],
      learningCurve: List[(Int, Double, Option[Double], Option[Double])]
  ) extends LoopState {
    def locations = List(
      model.location,
      optimizer.location
    ) ++ minValidationLossModel.toList.map(_._2.location)
  }
  case class SWALoopState(
      model: TensorList,
      optimizer: TensorList,
      epoch: Int,
      lastValidationLoss: Option[Double],
      minValidationLoss: Option[Double],
      numberOfAveragedModels: Int,
      averagedModels: Option[TensorList],
      learningCurve: List[(Int, Double, Option[Double])]
  ) extends LoopState {
    def locations = List(
      model.location,
      optimizer.location
    ) ++ averagedModels.toList.map(_.location)
  }
  case class SimpleThenSWALoopState(
      simple: SimpleLoopState,
      swa: Option[SWALoopState]
  ) extends LoopState {
    def locations: Seq[String] =
      simple.locations ++ swa.toList.flatMap(_.locations)
  }

  object LoopState {
    implicit val codec: JsonValueCodec[LoopState] = JsonCodecMaker.make
  }

  object LearningRateScheduleSchemas {
    implicit val codec
        : JsonValueCodec[nn.LearningRateSchedule.ReduceLROnPlateauState] =
      JsonCodecMaker.make
  }

}
