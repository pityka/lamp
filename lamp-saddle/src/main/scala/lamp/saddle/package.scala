package lamp

import org.saddle._
import lamp._
import lamp.STen._

package object saddle {

  /** Returns a tensor with the given content and shape on the given device */
  def fromMat[S: Sc](
      m: Mat[Double],
      cuda: Boolean = false
  ) = owned(SaddleTensorHelpers.fromMat(m, cuda))

  /** Returns a tensor with the given content and shape on the given device */
  def fromMat[S: Sc](
      m: Mat[Double],
      device: Device,
      precision: FloatingPointPrecision
  ) = owned(SaddleTensorHelpers.fromMat(m, device, precision))

  /** Returns a tensor with the given content and shape on the given device */
  def fromFloatMat[S: Sc](
      m: Mat[Float],
      device: Device
  ) = owned(SaddleTensorHelpers.fromFloatMat(m, device))

  /** Returns a tensor with the given content and shape on the given device */
  def fromVec[S: Sc](
      m: Vec[Double],
      cuda: Boolean = false
  ) = owned(SaddleTensorHelpers.fromVec(m, cuda))

  /** Returns a tensor with the given content and shape on the given device */
  def fromVec[S: Sc](
      m: Vec[Double],
      device: Device,
      precision: FloatingPointPrecision
  ) = if (m.isEmpty) STen.zeros(List(0), device.options(precision))
  else owned(SaddleTensorHelpers.fromVec(m, device, precision))

  /** Returns a tensor with the given content and shape on the given device */
  def fromLongMat[S: Sc](
      m: Mat[Long],
      device: Device
  ) = owned(SaddleTensorHelpers.fromLongMat(m, device))

  /** Returns a tensor with the given content and shape on the given device */
  def fromLongMat[S: Sc](
      m: Mat[Long],
      cuda: Boolean = false
  ) = owned(SaddleTensorHelpers.fromLongMat(m, cuda))

  /** Returns a tensor with the given content and shape on the given device */
  def fromLongVec[S: Sc](
      m: Vec[Long],
      device: Device
  ) = if (m.isEmpty) STen.zeros(List(0), device.to(STenOptions.l))
  else owned(SaddleTensorHelpers.fromLongVec(m, device))

  /** Returns a tensor with the given content and shape on the given device */
  def fromLongVec[S: Sc](
      m: Vec[Long],
      cuda: Boolean = false
  ) = owned(SaddleTensorHelpers.fromLongVec(m, cuda))

  implicit class StenSaddleSyntax(
      value: STen
  ) {

    /** Converts to a Mat[Double].
      *
      * Copies to CPU if needed. Fails if dtype is not float or double. Fails if
      * shape does not conform a matrix.
      */
    def toMat = SaddleTensorHelpers.toMat(value.value)

    /** Converts to a Mat[Float].
      *
      * Copies to CPU if needed. Fails if dtype is not float. Fails if shape
      * does not conform a matrix.
      */
    def toFloatMat = SaddleTensorHelpers.toFloatMat(value.value)

    /** Converts to a Mat[Long].
      *
      * Copies to CPU if needed. Fails if dtype is not long. Fails if shape does
      * not conform a matrix.
      */
    def toLongMat = SaddleTensorHelpers.toLongMat(value.value)

    /** Converts to a Vec[Double].
      *
      * Copies to CPU if needed. Fails if dtype is not float or double. Flattens
      * the shape.
      */
    def toVec = value.toDoubleArray.toVec

    /** Converts to a Vec[Float].
      *
      * Copies to CPU if needed. Fails if dtype is not float. Flattens the
      * shape.
      */
    def toFloatVec = value.toFloatArray.toVec

    /** Converts to a Vec[Long].
      *
      * Copies to CPU if needed. Fails if dtype is not long. Flattens the shape.
      */
    def toLongVec = value.toLongArray.toVec

  }

}
