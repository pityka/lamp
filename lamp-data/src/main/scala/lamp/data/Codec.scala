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
object IdentityCodec extends  Codec {
  def encode(in: Array[Byte]) = in.map(_.toChar)
  def decode(encoded: Array[Char]) = encoded.map(_.toByte)
  def saveToFile(file:File) : IO[Unit] = IO.unit
}
object IdentityCodecFactory extends CodecFactory[IdentityCodec.type] {
  def readFromFile(file:File) = IO.pure(IdentityCodec)
  def train(corpus:Array[Byte]) = IdentityCodec
}

