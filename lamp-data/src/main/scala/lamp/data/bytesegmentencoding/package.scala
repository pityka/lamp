package lamp.data

import java.io.File
import lamp.data.schemas.ByteSegmentEncoding

/** Greedy contraction of consecutive n-grams
  */
package object bytesegmentencoding {

  def saveEncodingToFile(
      file: File,
      encoding: Vector[(Vector[Byte], Char)]
  ): Unit = {
    val fos = new java.io.FileOutputStream(file)
    try {
      com.github.plokhotnyuk.jsoniter_scala.core
        .writeToStream(ByteSegmentEncoding(encoding), fos)
    } finally { fos.close }
  }
  def readEncodingFromFile(
      file: File
  ): Vector[(Vector[Byte], Char)] = {
    val fis = new java.io.FileInputStream(file)
    try {
      com.github.plokhotnyuk.jsoniter_scala.core
        .readFromStream[ByteSegmentEncoding](fis)
        .encoding
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

    val maxMergedSegmentLength = encoding.map(_._1.size).max
    val map = encoding.zipWithIndex.map(v => (v._1._1, (v._1._2, v._2))).toMap
    val n = corpus.length
    val output = Array.ofDim[Char](corpus.length)
    var outputCursor = 0
    var i = 0
    var j = 0
    while (i < n) {
      j = i + 1
      var encoded = unknownToken
      var priority = Int.MaxValue
      var usedJ = j
      while (j < i + maxMergedSegmentLength * 2 && j <= n) {
        val slice = corpus.slice(i, j).toVector
        val (a0, priority0) =
          map.get(slice).getOrElse((unknownToken, Int.MaxValue))
        if (priority0 < priority) {
          encoded = a0
          priority = priority0
          usedJ = slice.length
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
      maxMergedSegmentLength: Int
  ): Vector[(Vector[Byte], Char)] = {
    val frequencies = scala.collection.mutable.Map[Vector[Byte], Long]()
    var i = 0
    var j = 0
    val n = corpus.length
    while (i < n) {
      j = i + 1
      while (j <= i + maxMergedSegmentLength && j <= n) {
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
    frequencies.toVector
      .sortBy(v => -1 * v._2 * v._1.size)
      .take(vocabSize)
      .map(_._1)
      .zipWithIndex
      .map(v => (v._1, (v._2 + vocabularyMin).toChar))
  }

}
