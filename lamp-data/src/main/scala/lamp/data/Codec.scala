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

