package lamp.nnrf

import lamp._
import lamp.autograd._
import lamp.nn._
import scribe.Logger
import aten.ATen

import lamp.autograd.{Variable, param}
import aten.{ATen, TensorOptions}
import lamp.autograd.AllocatedVariablePool
import org.saddle._

case class Nnrf(cells: IndexedSeq[NnrfCell]) extends Module {
  override def state = cells.flatMap(_.state)
  val maxLevels = (math.log(cells.size) / math.log(2)).toInt
  def loop(
      level: Int,
      x: Variable,
      s1: Variable,
      acc: List[Variable]
  ): List[Variable] = {
    if (level > maxLevels) acc
    else {
      val cellsOnThisLevel = {
        val from = math.pow(2d, level).toInt - 1
        val to = math.pow(2d, level + 1).toInt - 1
        (from until to) toList
      }

      val levelOutputs = cellsOnThisLevel.zipWithIndex.map {
        case (cellIdx, levelIdx) =>
          val cell = cells(cellIdx)
          val s = s1.select(1, levelIdx).view(List(-1, 1))
          cell.forward((x, s))
      }
      val nextS = Concatenate(levelOutputs, 1).value.logSoftMax(1).exp
      loop(level + 1, x, nextS, nextS :: acc)

    }
  }
  def setData(data: Variable) = cells.foreach(_.setData(data))
  def forward(x: Variable): Variable = {
    val ones = const(
      TensorHelpers
        .fromMat(
          mat.ones(x.shape(0).toInt, 1),
          TensorHelpers.device(x.options),
          TensorHelpers.precision(x.value).get
        )
    )(x.pool).releasable
    val s1 = cells(0).forward((x, ones)).logSoftMax(1).exp
    val signals = loop(1, x, s1, Nil)
    val r = Concatenate(signals, 1).value
    // println(r.toMat.stringify(20, 20))
    r
  }
}

object Nnrf {
  implicit val trainingMode = TrainingMode.identity[Nnrf]
  implicit def load(implicit l: Load[NnrfCell]) = Load.make[Nnrf] {
    m => parameters =>
      implicit val pool = m.cells.head.weights.pool
      val groups = parameters.grouped(3).toList
      val loadedCells = (groups zip m.cells).map {
        case (params, cell) => cell.load(params)
      }

      m.copy(cells = loadedCells.toIndexedSeq)
  }
  def apply(
      levels: Int,
      numFeatures: Int,
      totalDataFeatures: Int,
      tOpt: TensorOptions
  )(implicit pool: AllocatedVariablePool): Nnrf =
    Nnrf(
      1 until math.pow(2d, levels).toInt map (i =>
        NnrfCell(
          numFeatures,
          totalDataFeatures,
          tOpt
        )
      )
    )
}

case class NnrfCell(weights: Variable, bias: Variable, features: Variable)
    extends GenericModule[
      (Variable, Variable),
      (Variable)
    ] {

  override val state = List(
    features -> NnrfCell.Features,
    weights -> NnrfCell.Weights,
    bias -> NnrfCell.Bias
  )

  var subsetX: Option[Variable] = None
  def subsetData(data: Variable) = {
    val fLong =
      const(ATen._cast_Long(features.value, false))(features.pool).releasable
    data.indexSelect(1, fLong).releaseWithVariable(fLong)
  }
  def setData(data: Variable) = {
    subsetX = Some(subsetData(data).keep)
  }

  def forward(tuple: (Variable, Variable)) = {
    val (x, s) = tuple

    (subsetX.getOrElse(subsetData(x)).mm(weights) + bias).relu * s
  }

}

object NnrfCell {

  implicit val trainingMode = TrainingMode.identity[NnrfCell]
  implicit val load = Load.make[NnrfCell] { m => parameters =>
    implicit val pool = m.weights.pool
    val f = param(parameters(0))
    val w = param(parameters(1))
    val b = param(parameters(2))
    m.copy(weights = w, bias = b, features = f)
  }
  case object Weights extends LeafTag
  case object Bias extends LeafTag
  case object Features extends PTag {
    def leaf: PTag = this
    def updateDuringOptimization: Boolean = false
  }
  def apply(
      numFeatures: Int,
      totalDataFeatures: Int,
      tOpt: TensorOptions
  )(implicit pool: AllocatedVariablePool): NnrfCell = {
    val in = numFeatures
    val out = 2
    NnrfCell(
      weights = param(
        ATen.normal_3(0d, math.sqrt(2d / in), Array(in, out), tOpt)
      ),
      bias = param(ATen.zeros(Array(1, out), tOpt)),
      features = param(
        TensorHelpers
          .fromVec(
            array
              .shuffle(array.range(0, totalDataFeatures))
              .take(numFeatures)
              .toVec
              .map(_.toDouble),
            TensorHelpers.device(tOpt),
            DoublePrecision
          )
      )
    )

  }

}
