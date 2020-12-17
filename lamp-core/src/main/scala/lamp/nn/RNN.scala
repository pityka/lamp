package lamp.nn

import lamp.autograd.{Variable, Constant, param}
import scala.collection.mutable
import lamp.Sc
import lamp.STen
import lamp.STenOptions

/** Inputs of size (sequence length * batch * in dim)
  * Outputs of size (sequence length * batch * hidden dim)
  */
case class RNN(
    weightXh: Constant,
    weightHh: Constant,
    biasH: Constant
) extends StatefulModule[Variable, Variable, Option[Variable]] {
  val inputSize = weightXh.shape.last
  val hiddenSize = biasH.shape.last

  override def state =
    List(
      (weightXh, RNN.WeightXh),
      (weightHh, RNN.WeightHh),
      (biasH, RNN.BiasH)
    )

  private def initHidden[S: Sc](batchSize: Long) = {
    param(STen.zeros(List(batchSize, hiddenSize), weightHh.options))
  }

  def forward[S: Sc](a: (Variable, Option[Variable])) = forward1(a._1, a._2)
  def forward1[S: Sc](x: Variable, state: Option[Variable]) = {
    val timesteps = x.shape.head
    val batchSize = x.shape(1)
    val outputs = mutable.ArrayBuffer[Variable]()
    val init = state.getOrElse(initHidden(batchSize))
    val lastHidden =
      (0 until timesteps.toInt).foldLeft(init) { (h, t) =>
        val xt = x.select(0, t)
        val newHidden = (xt.mm(weightXh) + h.mm(weightHh) + biasH).tanh

        outputs.append(newHidden)
        newHidden
      }
    (Variable.concatenateAddNewDim(outputs), Some(lastHidden))

  }

}

object RNN {
  implicit val trainingMode = TrainingMode.identity[RNN]
  implicit val is = InitState.make[RNN, Option[Variable]](_ => None)
  implicit val load = Load.make[RNN] { m => tensors =>
    m.weightXh.value.copyFrom(tensors(0))
    m.weightHh.value.copyFrom(tensors(1))
    m.biasH.value.copyFrom(tensors(2))

  }
  case object WeightXh extends LeafTag
  case object WeightHh extends LeafTag
  case object BiasH extends LeafTag

  def apply[S: Sc](
      in: Int,
      hiddenSize: Int,
      tOpt: STenOptions
  ): RNN =
    RNN(
      weightXh = param(
        STen.normal(
          0d,
          math.sqrt(2d / (in + hiddenSize)),
          List(in, hiddenSize),
          tOpt
        )
      ),
      weightHh = param(
        STen.normal(
          0d,
          math.sqrt(2d / (hiddenSize + hiddenSize)),
          List(hiddenSize, hiddenSize),
          tOpt
        )
      ),
      biasH = param(
        STen.zeros(
          List(1, hiddenSize),
          tOpt
        )
      )
    )

}
