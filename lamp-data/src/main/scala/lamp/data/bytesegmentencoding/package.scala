package lamp.data

import java.io.File
import lamp.data.schemas.ByteSegmentEncoding

/** Greedy contraction of consecutive n-grams
  */
package object bytesegmentencoding {

  def saveEncodingToFile(
      file: File,
      encoding: Vector[(Vector[Byte], Char)],
      unknownToken: Char,
      unknownByte: Byte
  ): Unit = {
    val fos = new java.io.FileOutputStream(file)
    try {
      com.github.plokhotnyuk.jsoniter_scala.core
        .writeToStream(
          ByteSegmentEncoding(
            encoding.map(v => (v._1, v._2.toInt)),
            unknownToken.toInt,
            unknownByte
          ),
          fos
        )
    } finally { fos.close }
  }
  def readEncodingFromFile(
      file: File
  ): ByteSegmentEncoding = {
    val fis = new java.io.FileInputStream(file)
    try {
      com.github.plokhotnyuk.jsoniter_scala.core
        .readFromStream[ByteSegmentEncoding](fis)
    } finally { fis.close }
  }

  def decode(
      encoded: Array[Char],
      encoding: Vector[(Vector[Byte], Char)],
      unknown: Byte
  ): Array[Byte] = {
    val map = encoding.map(_.swap).toMap
    encoded.flatMap(ch => map.get(ch).getOrElse(Vector(unknown)))
  }
  def encode(
      corpus: Array[Byte],
      encoding: Vector[(Vector[Byte], Char)],
      unknownToken: Char
  ): Array[Char] = {

    def pack(v: Array[Byte]): Long = {
      var l = 0L
      var i = 0
      while (i < v.length) {
        l |= ((v(i) & 255) << (8 * i))
        i += 1
      }
      l |= (v.length.toByte & 255) << 56
      l
    }
    def pack1(v: Byte): Long = {
      var l = 0L
      l |= v & 255
      l |= (1.toByte) << 56
      l
    }

    val maxMergedSegmentLength = encoding.map(_._1.size).max
    val map = scala.collection.mutable.LongMap(
      encoding.zipWithIndex
        .map(v => (pack(v._1._1.toArray), (v._1._2, v._2))): _*
    )

    val n = corpus.length
    val output = Array.ofDim[Char](corpus.length)
    var outputCursor = 0
    var i = 0
    var j = 0
    while (i < n) {
      j = i + 1
      var encoded = map.get(pack1(corpus(i))) match {
        case None         => unknownToken
        case Some((x, _)) => x
      }
      var priority = Int.MaxValue
      var usedJ = 1
      while (j < i + maxMergedSegmentLength && j <= n) {
        val slice = pack(corpus.slice(i, j))
        if (map.contains(slice)) {
          val (a0, priority0) =
            map(slice)
          if (priority0 < priority) {
            encoded = a0
            priority = priority0
            usedJ = j - i
          }
        }
        j += 1

      }
      output(outputCursor) = encoded
      outputCursor += 1
      i += usedJ
    }

    output.take(outputCursor)

  }

  /** Trains BPE encoding
    *
    * Char here is used as unsigned 16 bit integer
    *
    * @param corpus
    * @param vocabularyMin
    * @param vocabularyMax
    * @param maxMergedSegmentLength
    * @return
    */
  def train(
      corpus: Array[Byte],
      vocabularyMin: Char,
      vocabularyMax: Char,
      maxMergedSegmentLength: Int,
  ): Vector[(Vector[Byte], Char)] = {
    val effectiveMaxMergedSegmentLength = math.min(7,maxMergedSegmentLength)
    val frequencies = scala.collection.mutable.Map[Vector[Byte], Long]()
    var i = 0
    var j = 0
    val n = corpus.length
    while (i < n) {
      j = i + 1
      while (j <= i + effectiveMaxMergedSegmentLength && j <= n) {
        val sub = corpus.slice(i, j).toVector
          frequencies.get(sub) match {
            case None    => frequencies.update(sub, 1L)
            case Some(c) => frequencies.update(sub, c + 1L)
          }
        j += 1
      }
      i += 1
    }
    val vocabSize = vocabularyMax - vocabularyMin
    val singles = frequencies.keySet.toVector.filter(_.size == 1).distinct
    val nonSingles = frequencies.filter(v => v._1.size > 1 && v._1.forall(b => b.toChar.isLetterOrDigit))
    val r = (singles ++ nonSingles.toVector
      .sortBy(v => -1 * v._2)
      .take(vocabSize - singles.size)
      .map(_._1)).zipWithIndex
      .map(v => (v._1, (v._2 + vocabularyMin).toChar))
    assert(r.map(_._2).max <= vocabularyMax)
    assert(r.forall(_._1.size <= 7))
    r
  }

}
