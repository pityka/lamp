package lamp.nn

import lamp.autograd.{Variable, param, const}
import aten.Tensor
import aten.ATen
import scala.collection.mutable
import lamp.autograd.Concatenate
import aten.TensorOptions

/** Inputs of size (length * batch * vocab) */
case class RNN(
    weightXh: Variable,
    weightHh: Variable,
    weightHq: Variable,
    biasQ: Variable,
    biasH: Variable,
    var hiddenState: Variable,
    dropout: Double,
    train: Boolean
) extends Module {
  override def asEval: Module = copy(train = false)
  override def asTraining: Module = copy(train = true)

  override def load(tensors: Seq[Tensor]): Module = copy(
    weightXh = param(tensors(0)),
    weightHh = param(tensors(1)),
    weightHq = param(tensors(2)),
    biasQ = param(tensors(3)),
    biasH = param(tensors(4)),
    hiddenState = param(tensors(5))
  )

  override def state: Seq[(Variable, PTag)] = List(
    (weightXh, RNN.WeightXh),
    (weightHh, RNN.WeightHh),
    (weightHq, RNN.WeightHq),
    (biasQ, RNN.BiasQ),
    (biasH, RNN.BiasH),
    (hiddenState, RNN.State)
  )

  override def forward(x: Variable): Variable = {
    val timesteps = x.shape.head
    val outputs = mutable.ArrayBuffer[Variable]()
    val newHidden = (0 until timesteps.toInt).foldLeft(hiddenState) { (h, t) =>
      val xt = x.select(0, t)
      val newHidden =
        (xt.mm(weightXh) + h.releasable.mm(weightHh) + biasH).tanh.preserved
      val output = newHidden.dropout(dropout, train).mm(weightHq) + biasQ
      outputs.append(output)
      newHidden
    }
    hiddenState = newHidden
    Concatenate(outputs.toList, 0).value

  }

}

object RNN {
  case object WeightXh extends LeafTag
  case object WeightHh extends LeafTag
  case object WeightHq extends LeafTag
  case object BiasH extends LeafTag
  case object BiasQ extends LeafTag
  case object State extends IntermediateStateTag

  def apply(
      in: Int,
      hiddenSize: Int,
      out: Int,
      batchSize: Int,
      dropout: Double,
      tOpt: TensorOptions
  ): RNN =
    RNN(
      weightXh = param(
        ATen.normal_3(
          0d,
          math.sqrt(2d / (in + hiddenSize)),
          Array(hiddenSize, in),
          tOpt
        )
      ),
      weightHh = param(
        ATen.normal_3(
          0d,
          math.sqrt(2d / (hiddenSize + hiddenSize)),
          Array(hiddenSize, hiddenSize),
          tOpt
        )
      ),
      weightHq = param(
        ATen.normal_3(
          0d,
          math.sqrt(2d / (out + hiddenSize)),
          Array(out, hiddenSize),
          tOpt
        )
      ),
      biasQ = param(
        ATen.zeros(
          Array(1, out),
          tOpt
        )
      ),
      biasH = param(
        ATen.zeros(
          Array(1, hiddenSize),
          tOpt
        )
      ),
      hiddenState = param(
        ATen.normal_3(
          0d,
          math.sqrt(2d / (batchSize + hiddenSize)),
          Array(hiddenSize, batchSize),
          tOpt
        )
      ),
      dropout = dropout,
      train = true
    )

}
