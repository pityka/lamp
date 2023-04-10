package lamp.data

import java.io.File

trait CodecFactory[T<:Codec] {
  def readFromFile(file:File) : T 
  def train(corpus: Array[Byte]) : T 
}

trait Codec {
  def encode(in: Array[Byte]) : Array[Char]
  def decode(encoded: Array[Char]) : Array[Byte]
  def saveToFile(file:File) : Unit
}

object NoCodec extends Codec with CodecFactory[Codec] {

  def train(corpus: Array[Byte]) = NoCodec
  def encode(in: Array[Byte]) : Array[Char] = in.map(_.toChar)
  def decode(encoded: Array[Char]) : Array[Byte] = encoded.map(_.toByte)
  def saveToFile(file:File) : Unit = ()
  def readFromFile(file:File) : Codec = NoCodec
}