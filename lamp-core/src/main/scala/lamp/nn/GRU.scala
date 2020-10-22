package lamp.nn

import lamp.autograd.{Variable, param}
import aten.ATen
import scala.collection.mutable
import lamp.autograd.ConcatenateAddNewDim
import aten.TensorOptions
import lamp.Sc

/** Inputs of size (sequence length * batch * in dim)
  * Outputs of size (sequence length * batch * hidden dim)
  */
case class GRU(
    weightXh: Variable,
    weightHh: Variable,
    weightXr: Variable,
    weightXz: Variable,
    weightHr: Variable,
    weightHz: Variable,
    biasR: Variable,
    biasZ: Variable,
    biasH: Variable
) extends StatefulModule[Variable, Variable, Option[Variable]] {

  val inputSize = weightXh.shape.last
  val hiddenSize = biasH.shape.last

  override def state: Seq[(Variable, PTag)] =
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
    param(ATen.zeros(Array(batchSize, hiddenSize), weightHh.options))
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
    (ConcatenateAddNewDim(outputs).value, Some(lastHidden))

  }

}

object GRU {
  implicit val trainingMode = TrainingMode.identity[GRU]
  implicit val is = InitState.make[GRU, Option[Variable]](_ => None)
  implicit val load = Load.make[GRU] { m => tensors =>
    implicit val pool = m.weightHh.pool
    m.copy(
      weightXh = param(tensors(0)),
      weightHh = param(tensors(1)),
      weightXr = param(tensors(2)),
      weightXz = param(tensors(3)),
      weightHr = param(tensors(4)),
      weightHz = param(tensors(5)),
      biasR = param(tensors(6)),
      biasZ = param(tensors(7)),
      biasH = param(tensors(8))
    )
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
      )
    )

}
