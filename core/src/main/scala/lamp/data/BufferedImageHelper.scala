package lamp.data

import java.awt.image.BufferedImage
import lamp.autograd.TensorHelpers
import aten.Tensor
import aten.ATen
import aten.TensorOptions
import java.awt.Color
import lamp.syntax

object BufferedImageHelper {
  def fromFloatTensor(t: Tensor): BufferedImage = {
    val shape = t.shape
    assert(shape.size == 3, "Needs dim of 3")
    val arr = Array.ofDim[Float](shape.reduce(_ * _).toInt)
    assert(t.copyToFloatArray(arr))
    val stride = arr.size / 3
    var i = 0
    val bi = new BufferedImage(
      shape(1).toInt,
      shape(2).toInt,
      BufferedImage.TYPE_INT_ARGB
    )
    val width = shape(1).toInt
    while (i < stride) {
      val r = arr(i)
      val g = arr(i + stride)
      val b = arr(i + stride * 2)
      val color = new Color(r.toInt, g.toInt, b.toInt).getRGB()
      bi.setRGB(i % width, i / width, color)
      i += 1
    }

    bi
  }
  def fromDoubleTensor(t: Tensor): BufferedImage = {
    val shape = t.shape
    assert(shape.size == 3, "Needs dim of 3")
    val arr = Array.ofDim[Double](shape.reduce(_ * _).toInt)
    assert(t.copyToDoubleArray(arr))
    val stride = arr.size / 3
    var i = 0
    val bi = new BufferedImage(
      shape(1).toInt,
      shape(2).toInt,
      BufferedImage.TYPE_INT_ARGB
    )
    val width = shape(1).toInt
    while (i < stride) {
      val r = arr(i)
      val g = arr(i + stride)
      val b = arr(i + stride * 2)
      val color = new Color(r.toInt, g.toInt, b.toInt).getRGB()
      bi.setRGB(i % width, i / width, color)
      i += 1
    }

    bi
  }
  def toFloatTensor(image: BufferedImage): Tensor = {
    val dataAsInt =
      image.getRGB(
        0,
        0,
        image.getWidth(),
        image.getHeight(),
        null,
        0,
        image.getWidth
      )
    val ar = Array.ofDim[Float](dataAsInt.length * 3)
    val n = dataAsInt.length
    val r = {
      var i = 0
      while (i < n) {
        val c = dataAsInt(i)
        ar(i) = ((c & 0x00ff0000) >> 16).toFloat
        i += 1
      }
    }
    val g = {
      var i = n
      while (i < n * 2) {
        val c = dataAsInt(i - n)
        ar(i) = ((c & 0x0000ff00) >> 8).toFloat
        i += 1
      }
    }
    val b = {
      var i = n * 2
      while (i < n * 3) {
        val c = dataAsInt(i - n * 2)
        ar(i) = ((c & 0x000000ff)).toFloat
        i += 1
      }
    }
    val t = ATen.zeros(
      Array(3, image.getWidth, image.getHeight),
      TensorOptions.dtypeFloat
    )
    assert(t.copyFromFloatArray(ar), "Failed to copy")
    t
  }
}
