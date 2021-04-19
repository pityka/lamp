package lamp.nn

import lamp.autograd.{Variable, Constant, param, GraphConfiguration, GC}
import scala.collection.mutable
import lamp.Sc
import lamp.STen
import lamp.STenOptions

/** Inputs of size (sequence length * batch * vocab)
  * Outputs of size (sequence length * batch * output dim)
  */
case class LSTM(
    weightXi: Constant,
    weightXf: Constant,
    weightXo: Constant,
    weightHi: Constant,
    weightHf: Constant,
    weightHo: Constant,
    weightXc: Constant,
    weightHc: Constant,
    biasI: Constant,
    biasF: Constant,
    biasO: Constant,
    biasC: Constant,
    conf: GraphConfiguration
) extends StatefulModule[Variable, Variable, Option[(Variable, Variable)]] {
  val hiddenSize = biasI.shape.last

  override def state =
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

  private def initHidden[S: Sc](batchSize: Long) = {
    (
      param(STen.zeros(List(batchSize, hiddenSize), weightHf.options)),
      param(
        STen
          .zeros(List(batchSize, hiddenSize), weightHf.options)
      )
    )
  }

  def forward[S: Sc](a: (Variable, Option[(Variable, Variable)])) =
    forward1(a._1, a._2)
  private def forward1[S: Sc](
      x: Variable,
      state: Option[(Variable, Variable)]
  ) = {
    implicit def _conf = conf
    val timesteps = x.shape.head
    val batchSize = x.shape(1)
    val outputs = mutable.ArrayBuffer[Variable]()
    val init = state.getOrElse(initHidden(batchSize))
    val (lastHidden, lastMemory) =
      (0 until timesteps.toInt).foldLeft(init) { case ((h, c), t) =>
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
      Variable.concatenateAddNewDim(outputs.toSeq),
      Some((lastHidden, lastMemory))
    )

  }

}

object LSTM {
  implicit val trainingMode = TrainingMode.identity[LSTM]
  implicit val is =
    InitState.make[LSTM, Option[(Variable, Variable)]](_ => None)
  implicit val load = Load.make[LSTM] { m => tensors =>
    m.weightXi.value.copyFrom(tensors(0))
    m.weightXf.value.copyFrom(tensors(1))
    m.weightXo.value.copyFrom(tensors(2))
    m.weightHi.value.copyFrom(tensors(3))
    m.weightHf.value.copyFrom(tensors(4))
    m.weightHo.value.copyFrom(tensors(5))
    m.weightXc.value.copyFrom(tensors(6))
    m.weightHc.value.copyFrom(tensors(7))
    m.biasI.value.copyFrom(tensors(8))
    m.biasF.value.copyFrom(tensors(9))
    m.biasO.value.copyFrom(tensors(10))
    m.biasC.value.copyFrom(tensors(11))
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

  def apply[S: Sc, G: GC](
      in: Int,
      hiddenSize: Int,
      tOpt: STenOptions
  ): LSTM =
    LSTM(
      weightXi = param(
        STen.normal(
          0d,
          math.sqrt(2d / (in + hiddenSize)),
          List(in, hiddenSize),
          tOpt
        )
      ),
      weightXf = param(
        STen.normal(
          0d,
          math.sqrt(2d / (in + hiddenSize)),
          List(in, hiddenSize),
          tOpt
        )
      ),
      weightXo = param(
        STen.normal(
          0d,
          math.sqrt(2d / (in + hiddenSize)),
          List(in, hiddenSize),
          tOpt
        )
      ),
      weightXc = param(
        STen.normal(
          0d,
          math.sqrt(2d / (in + hiddenSize)),
          List(in, hiddenSize),
          tOpt
        )
      ),
      weightHi = param(
        STen.normal(
          0d,
          math.sqrt(2d / (hiddenSize + hiddenSize)),
          List(hiddenSize, hiddenSize),
          tOpt
        )
      ),
      weightHo = param(
        STen.normal(
          0d,
          math.sqrt(2d / (hiddenSize + hiddenSize)),
          List(hiddenSize, hiddenSize),
          tOpt
        )
      ),
      weightHf = param(
        STen.normal(
          0d,
          math.sqrt(2d / (hiddenSize + hiddenSize)),
          List(hiddenSize, hiddenSize),
          tOpt
        )
      ),
      weightHc = param(
        STen.normal(
          0d,
          math.sqrt(2d / (hiddenSize + hiddenSize)),
          List(hiddenSize, hiddenSize),
          tOpt
        )
      ),
      biasI = param(
        STen.zeros(
          List(1, hiddenSize),
          tOpt
        )
      ),
      biasF = param(
        STen.zeros(
          List(1, hiddenSize),
          tOpt
        )
      ),
      biasO = param(
        STen.zeros(
          List(1, hiddenSize),
          tOpt
        )
      ),
      biasC = param(
        STen.zeros(
          List(1, hiddenSize),
          tOpt
        )
      ),
      conf = implicitly[GraphConfiguration]
    )

}
