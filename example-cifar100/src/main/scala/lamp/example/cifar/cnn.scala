package lamp.example.cifar

import lamp._
import lamp.nn._
import aten.TensorOptions
import lamp.autograd.MaxPool2D
import lamp.autograd.Variable
import aten.Tensor
import lamp.util.NDArray
import lamp.autograd.AvgPool2D
import org.saddle.RankTie.Avg

case class Peek(label: String) extends Module {
  def state = Nil
  def forward(x: Variable): Variable = {
    scribe.info(s"PEEK - $label - ${x.shape}")
    x
  }

}

case class Residual[M1 <: Module, M2 <: Module](
    right: M1 with Module,
    left: Option[M2 with Module]
) extends Module {
  // override def asEval: Residual = copy(right.asEval, left.map(_.asEval))
  // override def asTraining: Residual =
  // copy(right.asTraining, left.map(_.asTraining))
  override def state = right.state ++ left.toList.flatMap(_.state)
  def forward(x: Variable) = {
    val r = right.forward(x)
    val l = left.map(_.forward(x)).getOrElse(x)
    (r + l)
  }

}

object Residual {
  implicit def trainingMode[M1 <: Module: Load, M2 <: Module: Load] =
    TrainingMode.identity[Residual[M1, M2]]
  implicit def load[M1 <: Module: Load, M2 <: Module: Load] =
    Load.make[Residual[M1, M2]](m =>
      t =>
        Residual(
          m.right.load(t.take(m.right.state.size)),
          m.left.map(l => l.load(t.drop(m.right.state.size).take(l.state.size)))
        )
    )
  def make(
      inChannels: Int,
      outChannels: Int,
      tOpt: TensorOptions,
      dropout: Double,
      stride: Int
  ) =
    sequence(
      Residual(
        right = Seq6(
          Conv2D(
            inChannels = inChannels,
            outChannels = outChannels,
            kernelSize = 3,
            padding = 1,
            stride = stride,
            tOpt = tOpt
          ),
          BatchNorm2D(outChannels, tOpt = tOpt),
          Fun(_.gelu),
          Dropout(dropout, true),
          Conv2D(
            inChannels = outChannels,
            outChannels = outChannels,
            kernelSize = 3,
            padding = 1,
            stride = 1,
            tOpt = tOpt
          ),
          BatchNorm2D(outChannels, tOpt = tOpt)
        ),
        left =
          if (inChannels == outChannels && stride == 1) None
          else
            Some(
              Seq2(
                Conv2D(
                  inChannels = inChannels,
                  outChannels = outChannels,
                  kernelSize = 1,
                  stride = stride,
                  padding = 0,
                  tOpt = tOpt
                ),
                BatchNorm2D(outChannels, tOpt = tOpt)
              )
            )
      ),
      Fun(_.gelu),
      Dropout(dropout, true)
    )

}

object Cnn {

  def resnet(
      width: Int,
      numClasses: Int,
      dropout: Double,
      tOpt: TensorOptions
  ) =
    sequence(
      Conv2D(
        inChannels = 3,
        outChannels = 6,
        kernelSize = 5,
        padding = 2,
        tOpt = tOpt
      ),
      sequence(
        Residual.make(
          inChannels = 6,
          outChannels = 6,
          tOpt = tOpt,
          dropout = dropout,
          stride = 2
        ),
        Residual.make(
          inChannels = 6,
          outChannels = 16,
          tOpt = tOpt,
          dropout = dropout,
          stride = 2
        ),
        Residual.make(
          inChannels = 16,
          outChannels = 128,
          tOpt = tOpt,
          dropout = dropout,
          stride = 1
        ),
        Residual.make(
          inChannels = 128,
          outChannels = numClasses,
          tOpt = tOpt,
          dropout = dropout,
          stride = 1
        )
      ),
      Fun(AvgPool2D(_, kernelSize = 8, padding = 0, stride = 1).value),
      Fun(_.flattenLastDimensions(3)),
      Fun(_.logSoftMax(dim = 1))
    )

  def lenet(
      numClasses: Int,
      dropOut: Double,
      tOpt: TensorOptions
  ) =
    Sequential(
      Conv2D(
        inChannels = 3,
        outChannels = 6,
        kernelSize = 5,
        padding = 2,
        tOpt = tOpt
      ),
      BatchNorm2D(6, tOpt),
      Fun(_.gelu),
      Dropout(dropOut, training = true),
      Fun(
        MaxPool2D(_, kernelSize = 2, stride = 2, padding = 0, dilation = 1).value
      ),
      Conv2D(
        inChannels = 6,
        outChannels = 16,
        kernelSize = 5,
        padding = 2,
        tOpt = tOpt
      ),
      BatchNorm2D(16, tOpt),
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
      Fun(_.logSoftMax(dim = 1))
    )
}
