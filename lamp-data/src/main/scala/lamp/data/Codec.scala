package lamp.data

import java.io.File
import cats.effect.IO

/** An abstraction around byte to token encodings.
  * 
  */
trait CodecFactory[T<:Codec] {
  def readFromFile(file:File) : IO[T] 
  def train(corpus: Array[Byte]) : T 
}

/** An abstraction around byte to token encodings.
  * 
  */
trait Codec {
  def encode(in: Array[Byte]) : Array[Char]
  def decode(encoded: Array[Char]) : Array[Byte]
  def saveToFile(file:File) : IO[Unit]
}

