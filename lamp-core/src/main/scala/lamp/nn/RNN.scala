package lamp.nn

import lamp.autograd.{Variable, param, const}
import aten.Tensor
import aten.ATen
import scala.collection.mutable
import lamp.autograd.ConcatenateAddNewDim
import aten.TensorOptions
import cats.effect.concurrent.Ref
import cats.effect.IO

/** Inputs of size (sequence length * batch * vocab)
  * Outputs of size (sequence length * batch * output dim)
  */
case class RNN(
    weightXh: Variable,
    weightHh: Variable,
    weightHq: Variable,
    biasQ: Variable,
    biasH: Variable,
    dropout: Double,
    train: Boolean
) extends StatefulModule[Option[Variable]] {
  override def asEval: RNN = copy(train = false)
  override def asTraining: RNN = copy(train = true)

  val inputSize = weightXh.shape.last
  val hiddenSize = biasH.shape.last
  val outputSize = biasQ.shape.last

  override def load(tensors: Seq[Tensor]): RNN = copy(
    weightXh = param(tensors(0)),
    weightHh = param(tensors(1)),
    weightHq = param(tensors(2)),
    biasQ = param(tensors(3)),
    biasH = param(tensors(4))
  )

  override def state: Seq[(Variable, PTag)] =
    List(
      (weightXh, RNN.WeightXh),
      (weightHh, RNN.WeightHh),
      (weightHq, RNN.WeightHq),
      (biasQ, RNN.BiasQ),
      (biasH, RNN.BiasH)
    )

  private def initHidden(batchSize: Long) = {
    param(ATen.zeros(Array(batchSize, hiddenSize), weightHh.options)).releasable
  }

  override def forward1(x: Variable, state: Option[Variable]) = {
    val timesteps = x.shape.head
    val batchSize = x.shape(1)
    val outputs = mutable.ArrayBuffer[Variable]()
    val init = state.getOrElse(initHidden(batchSize))
    val lastHidden =
      (0 until timesteps.toInt).foldLeft(init) { (h, t) =>
        val xt = x.select(0, t)
        val newHidden = (xt.mm(weightXh) + h.mm(weightHh) + biasH).tanh

        val output =
          (newHidden.dropout(dropout, train).mm(weightHq) + biasQ)

        outputs.append(output)
        newHidden
      }
    (ConcatenateAddNewDim(outputs).value, Some(lastHidden))

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
      dropout: Double,
      tOpt: TensorOptions
  ): RNN =
    RNN(
      weightXh = param(
        ATen.normal_3(
          0d,
          math.sqrt(2d / (in + hiddenSize)),
          Array(in, hiddenSize),
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
          Array(hiddenSize, out),
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
      dropout = dropout,
      train = true
    )

}
