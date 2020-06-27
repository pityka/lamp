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
case class LSTM(
    weightXi: Variable,
    weightXf: Variable,
    weightXo: Variable,
    weightHi: Variable,
    weightHf: Variable,
    weightHo: Variable,
    weightXc: Variable,
    weightHc: Variable,
    biasI: Variable,
    biasF: Variable,
    biasO: Variable,
    biasC: Variable
) extends StatefulModule[Variable, Variable, Option[(Variable, Variable)]] {
  val hiddenSize = biasI.shape.last

  override def state: Seq[(Variable, PTag)] =
    List(
      (weightXi, LSTM.WeightXi),
      (weightXf, LSTM.WeightXf),
      (weightXo, LSTM.WeightXo),
      (weightHi, LSTM.WeightXi),
      (weightHf, LSTM.WeightHf),
      (weightHo, LSTM.WeightHo),
      (weightXc, LSTM.WeightXc),
      (weightHc, LSTM.WeightHc),
      (biasI, LSTM.BiasI),
      (biasF, LSTM.BiasF),
      (biasO, LSTM.BiasO),
      (biasC, LSTM.BiasC)
    )

  private def initHidden(batchSize: Long) = {
    (
      param(ATen.zeros(Array(batchSize, hiddenSize), weightHf.options)).releasable,
      param(
        ATen
          .zeros(Array(batchSize, hiddenSize), weightHf.options)
      ).releasable
    )
  }

  def forward(a: (Variable, Option[(Variable, Variable)])) =
    forward1(a._1, a._2)
  private def forward1(x: Variable, state: Option[(Variable, Variable)]) = {
    val timesteps = x.shape.head
    val batchSize = x.shape(1)
    val outputs = mutable.ArrayBuffer[Variable]()
    val init = state.getOrElse(initHidden(batchSize))
    val (lastHidden, lastMemory) =
      (0 until timesteps.toInt).foldLeft(init) {
        case ((h, c), t) =>
          val xt = x.select(0, t)
          val it = (xt.mm(weightXi) + h.mm(weightHi) + biasI).sigmoid
          val ft = (xt.mm(weightXf) + h.mm(weightHf) + biasF).sigmoid
          val ot = (xt.mm(weightXo) + h.mm(weightHo) + biasO).sigmoid

          val ccap = (xt.mm(weightXc) + h.mm(weightHc) + biasC).tanh

          val ct = ft * c + it * ccap
          val ht = ot * ct.tanh

          outputs.append(ht)
          (ht, ct)
      }
    (
      ConcatenateAddNewDim(outputs).value,
      Some((lastHidden, lastMemory))
    )

  }

}

object LSTM {
  implicit val trainingMode = TrainingMode.identity[LSTM]
  implicit val is =
    InitState.make[LSTM, Option[(Variable, Variable)]](_ => None)
  implicit val load = Load.make[LSTM] { m => tensors =>
    m.copy(
      weightXi = param(tensors(0)),
      weightXf = param(tensors(1)),
      weightXo = param(tensors(2)),
      weightHi = param(tensors(3)),
      weightHf = param(tensors(4)),
      weightHo = param(tensors(5)),
      weightXc = param(tensors(6)),
      weightHc = param(tensors(7)),
      biasI = param(tensors(8)),
      biasF = param(tensors(9)),
      biasO = param(tensors(10)),
      biasC = param(tensors(11))
    )
  }
  case object WeightXi extends LeafTag
  case object WeightXf extends LeafTag
  case object WeightXo extends LeafTag
  case object WeightHi extends LeafTag
  case object WeightHf extends LeafTag
  case object WeightHo extends LeafTag
  case object WeightXc extends LeafTag
  case object WeightHc extends LeafTag
  case object BiasI extends LeafTag
  case object BiasF extends LeafTag
  case object BiasO extends LeafTag
  case object BiasC extends LeafTag

  def apply(
      in: Int,
      hiddenSize: Int,
      tOpt: TensorOptions
  ): LSTM =
    LSTM(
      weightXi = param(
        ATen.normal_3(
          0d,
          math.sqrt(2d / (in + hiddenSize)),
          Array(in, hiddenSize),
          tOpt
        )
      ),
      weightXf = param(
        ATen.normal_3(
          0d,
          math.sqrt(2d / (in + hiddenSize)),
          Array(in, hiddenSize),
          tOpt
        )
      ),
      weightXo = param(
        ATen.normal_3(
          0d,
          math.sqrt(2d / (in + hiddenSize)),
          Array(in, hiddenSize),
          tOpt
        )
      ),
      weightXc = param(
        ATen.normal_3(
          0d,
          math.sqrt(2d / (in + hiddenSize)),
          Array(in, hiddenSize),
          tOpt
        )
      ),
      weightHi = param(
        ATen.normal_3(
          0d,
          math.sqrt(2d / (hiddenSize + hiddenSize)),
          Array(hiddenSize, hiddenSize),
          tOpt
        )
      ),
      weightHo = param(
        ATen.normal_3(
          0d,
          math.sqrt(2d / (hiddenSize + hiddenSize)),
          Array(hiddenSize, hiddenSize),
          tOpt
        )
      ),
      weightHf = param(
        ATen.normal_3(
          0d,
          math.sqrt(2d / (hiddenSize + hiddenSize)),
          Array(hiddenSize, hiddenSize),
          tOpt
        )
      ),
      weightHc = param(
        ATen.normal_3(
          0d,
          math.sqrt(2d / (hiddenSize + hiddenSize)),
          Array(hiddenSize, hiddenSize),
          tOpt
        )
      ),
      biasI = param(
        ATen.zeros(
          Array(1, hiddenSize),
          tOpt
        )
      ),
      biasF = param(
        ATen.zeros(
          Array(1, hiddenSize),
          tOpt
        )
      ),
      biasO = param(
        ATen.zeros(
          Array(1, hiddenSize),
          tOpt
        )
      ),
      biasC = param(
        ATen.zeros(
          Array(1, hiddenSize),
          tOpt
        )
      )
    )

}
