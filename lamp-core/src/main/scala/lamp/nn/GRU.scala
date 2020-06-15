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
case class GRU(
    weightXh: Variable,
    weightHh: Variable,
    weightHq: Variable,
    weightXr: Variable,
    weightXz: Variable,
    weightHr: Variable,
    weightHz: Variable,
    biasR: Variable,
    biasZ: Variable,
    biasQ: Variable,
    biasH: Variable,
    dropout: Double,
    train: Boolean
) extends StatefulModule[Option[Variable]] {
  override def asEval: GRU = copy(train = false)
  override def asTraining: GRU = copy(train = true)

  val inputSize = weightXh.shape.last
  val hiddenSize = biasH.shape.last
  val outputSize = biasQ.shape.last

  override def load(tensors: Seq[Tensor]): GRU = copy(
    weightXh = param(tensors(0)),
    weightHh = param(tensors(1)),
    weightHq = param(tensors(2)),
    weightXr = param(tensors(3)),
    weightXz = param(tensors(4)),
    weightHr = param(tensors(5)),
    weightHz = param(tensors(6)),
    biasR = param(tensors(7)),
    biasZ = param(tensors(8)),
    biasQ = param(tensors(9)),
    biasH = param(tensors(10))
  )

  override def state: Seq[(Variable, PTag)] =
    List(
      (weightXh, GRU.WeightXh),
      (weightHh, GRU.WeightHh),
      (weightHq, GRU.WeightHq),
      (weightXr, GRU.WeightXr),
      (weightXz, GRU.WeightXz),
      (weightHr, GRU.WeightHr),
      (weightHz, GRU.WeightHz),
      (biasR, GRU.BiasR),
      (biasZ, GRU.BiasZ),
      (biasQ, GRU.BiasQ),
      (biasH, GRU.BiasH)
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
        val r = (xt.mm(weightXr) + h.mm(weightHr) + biasR).sigmoid
        val z = (xt.mm(weightXz) + h.mm(weightHz) + biasZ).sigmoid
        val hcap = (xt.mm(weightXh) + (r.*(h)).mm(weightHh) + biasH).tanh

        val newHidden = z * h + ((z.*(-1)).+(1d)).*(hcap)

        val output =
          (newHidden.dropout(dropout, train).mm(weightHq) + biasQ)

        outputs.append(output)
        newHidden
      }
    (ConcatenateAddNewDim(outputs).value, Some(lastHidden))

  }

}

object GRU {
  case object WeightXh extends LeafTag
  case object WeightHh extends LeafTag
  case object WeightHq extends LeafTag
  case object WeightHr extends LeafTag
  case object WeightHz extends LeafTag
  case object WeightXr extends LeafTag
  case object WeightXz extends LeafTag
  case object BiasR extends LeafTag
  case object BiasZ extends LeafTag
  case object BiasH extends LeafTag
  case object BiasQ extends LeafTag

  def apply(
      in: Int,
      hiddenSize: Int,
      out: Int,
      dropout: Double,
      tOpt: TensorOptions
  ): GRU =
    GRU(
      weightXh = param(
        ATen.normal_3(
          0d,
          math.sqrt(2d / (in + hiddenSize)),
          Array(in, hiddenSize),
          tOpt
        )
      ),
      weightXr = param(
        ATen.normal_3(
          0d,
          math.sqrt(2d / (in + hiddenSize)),
          Array(in, hiddenSize),
          tOpt
        )
      ),
      weightXz = param(
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
      weightHr = param(
        ATen.normal_3(
          0d,
          math.sqrt(2d / (hiddenSize + hiddenSize)),
          Array(hiddenSize, hiddenSize),
          tOpt
        )
      ),
      weightHz = param(
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
      biasR = param(
        ATen.zeros(
          Array(1, hiddenSize),
          tOpt
        )
      ),
      biasZ = param(
        ATen.zeros(
          Array(1, hiddenSize),
          tOpt
        )
      ),
      dropout = dropout,
      train = true
    )

}
