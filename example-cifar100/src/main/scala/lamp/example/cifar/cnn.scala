package lamp.example.cifar

import lamp._
import lamp.nn._
import aten.TensorOptions
import lamp.autograd.MaxPool2D
import lamp.autograd.Variable
import aten.Tensor
import lamp.util.NDArray
import lamp.autograd.AvgPool2D

case class Peek(label: String) extends Module {

  def forward(x: Variable): Variable = {
    scribe.info(s"PEEK - $label - ${x.shape}")
    x
  }

}

object Cnn {

  def resnet(
      width: Int,
      height: Int,
      numClasses: Int,
      dropOut: Double,
      tOpt: TensorOptions
  ) = Sequential(
    residualBlock(3, 32, 16, tOpt),
    residualBlock(32, 32, 16, tOpt),
    Fun(v =>
      MaxPool2D(v, kernelSize = 2, stride = 2, padding = 0, dilation = 1).value
    ),
    residualBlock(32, 32, 16, tOpt),
    Fun(v =>
      MaxPool2D(v, kernelSize = 2, stride = 2, padding = 0, dilation = 1).value
    ),
    residualBlock(32, 32, 16, tOpt),
    gap(
      kernelSize = width / 4,
      inChannels = 32,
      numClasses = numClasses,
      tOpt
    )
  )

  def gap(
      kernelSize: Int,
      inChannels: Int,
      numClasses: Int,
      tOpt: TensorOptions
  ) = Sequential(
    residualBlock(inChannels, numClasses, 16, tOpt),
    Fun(v => AvgPool2D(v, kernelSize, stride = 1, padding = 0).value),
    Fun(_.flattenLastDimensions(3)),
    Fun(_.logSoftMax)
  )

  def residualBlock(
      inChannels: Int,
      outChannels: Int,
      internalChannels: Int,
      tOpt: TensorOptions
  ) = {
    val mod = Sequential(
      Conv2D(
        inChannels = inChannels,
        outChannels = internalChannels,
        kernelSize = 1,
        padding = 0,
        stride = 1,
        tOpt = tOpt
      ),
      BatchNorm2D(internalChannels, tOpt = tOpt),
      Fun(_.gelu),
      Conv2D(
        inChannels = internalChannels,
        outChannels = internalChannels,
        kernelSize = 3,
        stride = 1,
        padding = 1,
        groups = math.ceil(internalChannels / 512d).toLong,
        tOpt = tOpt
      ),
      BatchNorm2D(internalChannels, tOpt = tOpt),
      Fun(_.gelu),
      Conv2D(
        inChannels = internalChannels,
        outChannels = outChannels,
        kernelSize = 1,
        stride = 1,
        padding = 0,
        groups = math.ceil(internalChannels / 512d).toLong,
        tOpt = tOpt
      ),
      BatchNorm2D(outChannels, tOpt = tOpt)
    )
    if (inChannels == outChannels)
      Sequential(
        Residual(
          mod
        ),
        Fun(_.gelu)
      )
    else mod
  }

  def residual(v: Variable)(f: Variable => Variable): Variable = v + f(v)

  def lenet(
      numClasses: Int,
      dropOut: Double,
      tOpt: TensorOptions
  ) =
    Sequential(
      // Peek,
      Conv2D(
        inChannels = 3,
        outChannels = 6,
        kernelSize = 5,
        padding = 2,
        tOpt = tOpt
      ),
      BatchNorm2D(6, tOpt),
      // Peek,
      Fun(_.gelu),
      Dropout(dropOut, training = true),
      Fun(
        MaxPool2D(_, kernelSize = 2, stride = 2, padding = 0, dilation = 1).value
      ),
      // Peek,
      Conv2D(
        inChannels = 6,
        outChannels = 16,
        kernelSize = 5,
        padding = 2,
        tOpt = tOpt
      ),
      BatchNorm2D(16, tOpt),
      // Peek,
      Fun(_.gelu),
      Dropout(dropOut, training = true),
      Fun(
        MaxPool2D(_, kernelSize = 2, stride = 2, padding = 0, dilation = 1).value
      ),
      Fun(_.flattenLastDimensions(3)),
      Linear(1024, 120, tOpt = tOpt),
      BatchNorm(120, tOpt),
      Fun(_.gelu),
      Dropout(dropOut, training = true),
      Linear(120, 84, tOpt = tOpt),
      BatchNorm(84, tOpt),
      Fun(_.gelu),
      Dropout(dropOut, training = true),
      Linear(84, numClasses, tOpt = tOpt),
      Fun(_.logSoftMax)
    )
}
