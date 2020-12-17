package lamp.nn

import lamp.autograd.{Variable, Constant, param}
import scala.collection.mutable
import lamp.STenOptions
import lamp.Sc
import lamp.STen

/** Inputs of size (sequence length * batch * in dim)
  * Outputs of size (sequence length * batch * hidden dim)
  */
case class GRU(
    weightXh: Constant,
    weightHh: Constant,
    weightXr: Constant,
    weightXz: Constant,
    weightHr: Constant,
    weightHz: Constant,
    biasR: Constant,
    biasZ: Constant,
    biasH: Constant
) extends StatefulModule[Variable, Variable, Option[Variable]] {

  val inputSize = weightXh.shape.last
  val hiddenSize = biasH.shape.last

  override def state =
    List(
      (weightXh, GRU.WeightXh),
      (weightHh, GRU.WeightHh),
      (weightXr, GRU.WeightXr),
      (weightXz, GRU.WeightXz),
      (weightHr, GRU.WeightHr),
      (weightHz, GRU.WeightHz),
      (biasR, GRU.BiasR),
      (biasZ, GRU.BiasZ),
      (biasH, GRU.BiasH)
    )

  private def initHidden[S: Sc](batchSize: Long) = {
    param(STen.zeros(List(batchSize, hiddenSize), weightHh.options))
  }
  def initState = None
  def forward[S: Sc](a: (Variable, Option[Variable])) = forward1(a._1, a._2)
  private def forward1[S: Sc](x: Variable, state: Option[Variable]) = {
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

        outputs.append(newHidden)
        newHidden
      }
    (Variable.concatenateAddNewDim(outputs), Some(lastHidden))

  }

}

object GRU {
  implicit val trainingMode = TrainingMode.identity[GRU]
  implicit val is = InitState.make[GRU, Option[Variable]](_ => None)
  implicit val load = Load.make[GRU] { m => tensors =>
    m.weightXh.value.copyFrom(tensors(0))
    m.weightHh.value.copyFrom(tensors(1))
    m.weightXr.value.copyFrom(tensors(2))
    m.weightXz.value.copyFrom(tensors(3))
    m.weightHr.value.copyFrom(tensors(4))
    m.weightHz.value.copyFrom(tensors(5))
    m.biasR.value.copyFrom(tensors(6))
    m.biasZ.value.copyFrom(tensors(7))
    m.biasH.value.copyFrom(tensors(8))

  }
  case object WeightXh extends LeafTag
  case object WeightHh extends LeafTag
  case object WeightHr extends LeafTag
  case object WeightHz extends LeafTag
  case object WeightXr extends LeafTag
  case object WeightXz extends LeafTag
  case object BiasR extends LeafTag
  case object BiasZ extends LeafTag
  case object BiasH extends LeafTag

  def apply[S: Sc](
      in: Int,
      hiddenSize: Int,
      tOpt: STenOptions
  ): GRU =
    GRU(
      weightXh = param(
        STen.normal(
          0d,
          math.sqrt(2d / (in + hiddenSize)),
          List(in, hiddenSize),
          tOpt
        )
      ),
      weightXr = param(
        STen.normal(
          0d,
          math.sqrt(2d / (in + hiddenSize)),
          List(in, hiddenSize),
          tOpt
        )
      ),
      weightXz = param(
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
      weightHr = param(
        STen.normal(
          0d,
          math.sqrt(2d / (hiddenSize + hiddenSize)),
          List(hiddenSize, hiddenSize),
          tOpt
        )
      ),
      weightHz = param(
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
      ),
      biasR = param(
        STen.zeros(
          List(1, hiddenSize),
          tOpt
        )
      ),
      biasZ = param(
        STen.zeros(
          List(1, hiddenSize),
          tOpt
        )
      )
    )

}
