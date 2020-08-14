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

case class Nnrf(
    levels: IndexedSeq[NnrfCell],
    heads: Seq[Variable],
    biases: Seq[Variable]
) extends Module {

  override def state =
    levels.flatMap(_.state) ++ heads.map(v => (v -> NoTag)) ++ biases.map(v =>
      (v -> NoTag)
    )
  val maxLevels = levels.size
  def loop(
      level: Int,
      x: Variable,
      s1: Variable,
      acc: List[Variable]
  ): List[Variable] = {
    if (level >= maxLevels) acc
    else {
      val r = levels(level).forward((x, s1))
      val nextS =
        r.logSoftMax(2)
          .exp
          .transpose(1, 2)
          .reshape(List(-1, r.shape(1).toInt, 1))
      loop(level + 1, x, nextS, nextS :: acc)

    }
  }
  def setData(data: Variable) = levels.foreach(_.setData(data))
  def forward(x: Variable): Variable = {
    val ones = const(
      TensorHelpers
        .fromMat(
          mat.ones(x.shape(0).toInt, 1),
          TensorHelpers.device(x.options),
          TensorHelpers.precision(x.value).get
        )
    )(x.pool).releasable.view(List(1, -1, 1))
    val samples = x.shape(0).toInt
    val signals = loop(0, x, ones, Nil)
    val r = signals.reverse zip heads zip biases map {
      case ((s, h), b) =>
        s.view(List(-1, samples)).transpose(0, 1) mm h + b
    } reduce (_ + _)

    r
  }
}

object Nnrf {
  implicit val trainingMode = TrainingMode.identity[Nnrf]
  // implicit def load(implicit l: Load[NnrfCell]) = Load.make[Nnrf] {
  //   m => parameters =>
  //     implicit val pool = m.cells.head.weights.pool
  //     val groups = parameters.grouped(3).toList
  //     val loadedCells = (groups zip m.cells).map {
  //       case (params, cell) => cell.load(params)
  //     }

  //     m.copy(cells = loadedCells.toIndexedSeq)
  // }
  def apply(
      levels: Int,
      numFeatures: Int,
      totalDataFeatures: Int,
      out: Int,
      tOpt: TensorOptions
  )(implicit pool: AllocatedVariablePool): Nnrf =
    Nnrf(
      0 until levels.toInt map (i =>
        NnrfCell(
          math.pow(2d, i).toInt,
          numFeatures,
          totalDataFeatures,
          tOpt
        )
      ),
      0 until levels map { i =>
        val in = math.pow(2d, i + 1).toInt
        param(
          ATen.normal_3(0d, math.sqrt(2d / in), Array(in, out), tOpt)
        )
      },
      0 until levels map { i => param(ATen.zeros(Array(1, out), tOpt)) }
    )
}

// Trees x Samples x features
case class NnrfCell(
    weights: Variable,
    biases: Variable,
    features: Variable,
    featuresPerNode: Int
) extends GenericModule[
      (Variable, Variable),
      (Variable)
    ] {

  override val state = List(
    features -> NnrfCell.Features,
    weights -> NnrfCell.Weights,
    biases -> NnrfCell.Bias
  )

  var subsetX: Option[Variable] = None
  def subsetData(data: Variable) = {
    val fLong =
      const(ATen._cast_Long(features.value, false))(features.pool).releasable
    data
      .indexSelect(1, fLong)
      .releaseWithVariable(fLong)
      .view(List(data.shape(0).toInt, -1, featuresPerNode))
      .transpose(0, 1)
  }
  def setData(data: Variable) = {
    subsetX = Some(subsetData(data).keep)
  }

  def forward(tuple: (Variable, Variable)) = {
    val (x, s) = tuple
    val features = subsetX.getOrElse(subsetData(x))
    val r = (features.bmm(weights) + biases).relu
    r * s
  }

}

object NnrfCell {

  implicit val trainingMode = TrainingMode.identity[NnrfCell]
  implicit val load = Load.make[NnrfCell] { m => parameters =>
    implicit val pool = m.weights.pool
    val f = param(parameters(0))
    val w = param(parameters(1))
    val b = param(parameters(2))
    m.copy(weights = w, biases = b, features = f)
  }
  case object Weights extends LeafTag
  case object Bias extends LeafTag
  case object Features extends PTag {
    def leaf: PTag = this
    def updateDuringOptimization: Boolean = false
  }
  def apply(
      nodes: Int,
      numFeatures: Int,
      totalDataFeatures: Int,
      tOpt: TensorOptions
  )(implicit pool: AllocatedVariablePool): NnrfCell = {
    val in = numFeatures
    val out = 2
    NnrfCell(
      weights = param(
        ATen.normal_3(0d, math.sqrt(2d / in), Array(nodes, in, out), tOpt)
      ),
      biases = param(ATen.zeros(Array(nodes, 1, out), tOpt)),
      features = param(
        TensorHelpers
          .fromVec(
            (0 until nodes flatMap (_ =>
              array
                .shuffle(array.range(0, totalDataFeatures))
                .take(numFeatures)
            )).toVec
              .map(_.toDouble),
            TensorHelpers.device(tOpt),
            DoublePrecision
          )
      ),
      featuresPerNode = numFeatures
    )

  }

}
