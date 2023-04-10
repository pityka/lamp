package lamp.data.bytesegmentencoding

import java.io.File

case class ByteSegmentCodec(
    trained: Vector[(Vector[Byte], Char)],
    unknownToken: Char,
    unknownByte: Byte
) extends lamp.data.Codec {
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
  def saveToFile(file: File): Unit = lamp.data.bytesegmentencoding
    .saveEncodingToFile(file, trained, unknownToken, unknownByte)

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

  def readFromFile(file: File): ByteSegmentCodec = {
    val r = lamp.data.bytesegmentencoding.readEncodingFromFile(file)
    ByteSegmentCodec(
      r.encoding,
      r.unknownToken,
      r.unknownByte
    )
  }
}
