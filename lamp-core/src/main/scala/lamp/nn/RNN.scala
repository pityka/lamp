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
    hiddenState: Ref[IO, Variable],
    dropout: Double,
    train: Boolean
) extends Module {
  override def asEval: Module = copy(train = false)
  override def asTraining: Module = copy(train = true)

  val inputSize = weightXh.shape.last
  val hiddenSize = biasH.shape.last
  val outputSize = biasQ.shape.last
  val batchSize = hiddenState.get.map(_.shape.head).unsafeRunSync()

  override def load(tensors: Seq[Tensor]): Module = copy(
    weightXh = param(tensors(0)),
    weightHh = param(tensors(1)),
    weightHq = param(tensors(2)),
    biasQ = param(tensors(3)),
    biasH = param(tensors(4)),
    hiddenState = Ref.of[IO, Variable](param(tensors(5))).unsafeRunSync()
  )

  override def state: Seq[(Variable, PTag)] =
    hiddenState.get
      .map { hiddenState =>
        List(
          (weightXh, RNN.WeightXh),
          (weightHh, RNN.WeightHh),
          (weightHq, RNN.WeightHq),
          (biasQ, RNN.BiasQ),
          (biasH, RNN.BiasH),
          (hiddenState, RNN.State)
        )
      }
      .unsafeRunSync()

  override def forward(x: Variable): Variable = {
    hiddenState.get
      .flatMap { hiddenState =>
        val timesteps = x.shape.head
        val outputs = mutable.ArrayBuffer[Variable]()
        val newHidden =
          (0 until timesteps.toInt).foldLeft(hiddenState.releasable) { (h, t) =>
            val xt = x.select(0, t)
            val newHidden = {
              val v = (xt.mm(weightXh) + h.mm(weightHh) + biasH).tanh
              if (t == timesteps.toInt - 1) v.preserved
              else v
            }
            val output =
              (newHidden.dropout(dropout, train).mm(weightHq) + biasQ)

            outputs.append(output)
            newHidden
          }
        this.hiddenState
          .set(newHidden.detached)
          .map(_ => ConcatenateAddNewDim(outputs).value)

      }
      .unsafeRunSync()

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
      hiddenState = Ref
        .of[IO, Variable](
          param(
            ATen.normal_3(
              0d,
              math.sqrt(2d / (batchSize + hiddenSize)),
              Array(batchSize, hiddenSize),
              tOpt
            )
          )
        )
        .unsafeRunSync(),
      dropout = dropout,
      train = true
    )

}
