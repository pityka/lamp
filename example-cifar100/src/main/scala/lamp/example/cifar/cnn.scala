package lamp.example.cifar

import lamp._
import lamp.nn._
import aten.TensorOptions
import lamp.autograd.MaxPool2D
import lamp.autograd.Variable
import aten.Tensor
import lamp.util.NDArray

case object Peek extends Module {

  def forward(x: Variable): Variable = {
    scribe.info(s"PEEK - ${x.shape}")
    x
  }

}

object Cnn {
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
      // Peek,
      Fun(_.gelu),
      Dropout(dropOut, training = true),
      Fun(
        MaxPool2D(_, kernelSize = 2, stride = 2, padding = 0, dilation = 1).value
      ),
      FlattenLast(3),
      Linear(1024, 120, tOpt = tOpt),
      Fun(_.gelu),
      Dropout(dropOut, training = true),
      Linear(120, 84, tOpt = tOpt),
      Fun(_.gelu),
      Dropout(dropOut, training = true),
      Linear(84, numClasses, tOpt = tOpt),
      Fun(_.logSoftMax)
    )
}

// case class Mlp1(dim: Int, k: Int, y: Variable) extends Module {

//   def load(parameters: Seq[Tensor]) = this

//   val mod = Sequential(
//     Linear(
//       param(ATen.ones(Array(32, dim), y.options)),
//       Some(param(ATen.ones(Array(1, 32), y.options)))
//     ),
//     Fun(_.logSoftMax),
//     Fun(_.gelu),
//     Linear(
//       param(ATen.ones(Array(k, 32), y.options)),
//       Some(param(ATen.ones(Array(1, k), y.options)))
//     )
//   )

//   def forward(x: Variable): Variable =
//     mod.forward(x).crossEntropy(y).sum +
//       mod.parameters
//         .map(_._1.squaredFrobenius)
//         .reduce(_ + _)

//   def parameters: Seq[(Variable, PTag)] =
//     mod.parameters

// }
