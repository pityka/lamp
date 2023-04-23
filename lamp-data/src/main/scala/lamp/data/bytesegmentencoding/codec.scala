package lamp.data.bytesegmentencoding

import java.io.File
import cats.effect.IO

case class ByteSegmentCodec(
    trained: Vector[(Vector[Byte], Char)],
    unknownToken: Char,
    unknownByte: Byte
) extends lamp.data.Codec {

  override def toString() = s"ByteSegmentCodec(unknownToken=${unknownToken.toInt},unknownByte=$unknownByte,trained(top1000)=${trained.map(v => v._1.map(_.toChar).mkString -> v._2.toInt).take(1000).mkString("\n","\n","\n")})"
  def encode(in: Array[Byte]): Array[Char] =
    lamp.data.bytesegmentencoding.encode(
      corpus = in,
      encoding = trained,
      unknownToken = unknownToken
    )
  def decode(encoded: Array[Char]): Array[Byte] =
    lamp.data.bytesegmentencoding.decode(
      encoded,
      trained,
      unknownByte
    )
  def saveToFile(file: File): IO[Unit] = IO.blocking{
    lamp.data.bytesegmentencoding
    .saveEncodingToFile(file, trained, unknownToken, unknownByte)
  }

}
case class ByteSegmentCodecFactory(
    vocabularyMin: Char,
    vocabularyMax: Char,
    maxMergedSegmentLength: Int,
    unknownToken: Char,
    unknownByte: Byte
) extends lamp.data.CodecFactory[ByteSegmentCodec] {
  def train(
      corpus: Array[Byte]
  ) =
    ByteSegmentCodec(
      lamp.data.bytesegmentencoding
        .train(corpus, vocabularyMin, vocabularyMax, maxMergedSegmentLength),
      unknownToken,
      unknownByte
    )

  def readFromFile(file: File): IO[ByteSegmentCodec] = IO.blocking{
    val r = lamp.data.bytesegmentencoding.readEncodingFromFile(file)
    ByteSegmentCodec(
      r.encoding.map(v => (v._1, v._2.toChar)),
      r.unknownToken.toChar,
      r.unknownByte
    )
  }
}
