package lamp.data.safetensors

import java.io.File
import java.io.FileInputStream
import lamp.data.Reader.readFully
import java.nio.ByteBuffer
import java.nio.ByteOrder
import lamp.STen
import lamp.Device
import lamp.Scope

/** Parsers files written by
  * https://github.com/huggingface/safetensors/tree/752c1ab3b52463f4c4efda056e4c6a41e81a7ff3
  */
object SafeTensorReader {
  case class TensorList(
      tensors: Map[String, STen],
      meta: Map[String, Option[String]]
  )

  private object Schema {
    case class TensorDescriptor(
        dtype: String,
        shape: Seq[Long],
        offsets: (Long, Long)
    )
    case class TensorListDescriptor(
        tensors: Map[String, TensorDescriptor],
        meta: Map[String, Option[String]]
    )

    def parse(a: Array[Byte]): TensorListDescriptor = {
      val parsed = com.github.plokhotnyuk.jsoniter_scala.core
        .readFromArray[dijon.SomeJson](a)
        .toMap
      val meta = parsed
        .get("__metadata__")
        .map(_.toMap.map { case (a, b) =>
          (a, b.asString)
        })
        .getOrElse(Map.empty)
      def asLong(x: dijon.SomeJson) =
        x.asDouble.map(_.round).getOrElse(x.asInt.get.toLong)
      val tensors = parsed
        .filterNot(_._1 == "__metadata__")
        .map { case (name, v) =>
          name -> TensorDescriptor(
            dtype = v.apply("dtype").asString.get,
            shape = v.apply("shape").toSeq.map(x => asLong(x)).toList,
            offsets = {
              val k = v.apply("data_offsets").toSeq.map(asLong)
              (k(0), k(1))
            }
          )
        }
        .toMap
      TensorListDescriptor(tensors, meta.toMap)

    }

    def mapDType(s: String): Byte = s.toLowerCase.strip match {
      case "bool" => 11
      case "u8"   => 0
      case "i8"   => 1
      case "i16"  => 2
      case "i32"  => 3
      case "i64"  => 4
      case "f16"  => 5
      case "f32"  => 6
      case "f64"  => 7
      case "bf16" => 15
    }
  }

  def read(file: File, device: Device)(implicit
      scope: Scope
  ): TensorList = {
    val channel = new FileInputStream(file).getChannel()
    val headerLengthBB =
      ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN)
    readFully(headerLengthBB, channel)
    val headerLength = headerLengthBB.getLong.toInt
    require(headerLength % 8 == 0, "header length must be multiple of 8")
    val headerArray = Array.ofDim[Byte](headerLength)
    readFully(ByteBuffer.wrap(headerArray), channel)
    val descriptor = Schema.parse(headerArray)
    val list = descriptor.tensors.toVector.sortBy(_._1)

    val tensors = STen
      .tensorsFromFile(
        path = file.getAbsolutePath(),
        offset = 0L,
        length = file.length(),
        pin = false,
        tensors = list.map { case (_, td) =>
          (
            Schema.mapDType(td.dtype),
            td.offsets._1 + 8 + headerLength,
            td.offsets._2 - td.offsets._1
          )
        }.toList
      )
      .zip(list)
      .map { case (t1, (name, descr)) =>
        (name, device.to(t1.view(descr.shape: _*)))
      }
      .toMap

    TensorList(
      tensors = tensors,
      meta = descriptor.meta
    )
  }
}
