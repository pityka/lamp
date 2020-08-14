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
import aten.Tensor

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

  def predict(
      features: Mat[Double],
      model: Seq2[Variable, Variable, Variable, Nnrf, Fun]
  ) = {
    implicit val pool = model.m1.heads.head.pool
    val device = TensorHelpers.device(model.m1.biases.head.value)
    val precision = TensorHelpers.precision(model.m1.biases.head.value).get
    val x =
      const(
        TensorHelpers
          .fromMat(features, device, precision)
      ).releasable
    val output = model.forward(x)
    val outputJ = output.toMat
    output.releaseAll()
    pool.releaseAll
    outputJ
  }

  def trainClassification(
      features: Mat[Double],
      target: Vec[Long],
      numClasses: Int,
      classWeights: Vec[Double],
      device: Device,
      precision: FloatingPointPrecision = SinglePrecision,
      learningRate: Double = 0.001,
      epochs: Int = 300,
      logger: Option[scribe.Logger] = None
  ) = {
    implicit val pool = new AllocatedVariablePool
    val targetT = ATen.squeeze_0(
      TensorHelpers.fromLongMat(
        Mat(target),
        device
      )
    )
    val x =
      const(
        TensorHelpers
          .fromMat(features, device, precision)
      )
    val tOpt = device.options(precision)
    val model = Seq2(
      Nnrf.apply(
        levels = 5,
        numFeatures = 32,
        totalDataFeatures = 784,
        out = 10,
        tOpt = tOpt,
        fullBatchData = Some(x)
      ),
      Fun(_.logSoftMax(dim = 1))
    )
    val optim = AdamW(
      model.parameters.map(v => (v._1.value, v._2)),
      learningRate = simple(0.001),
      weightDecay = simple(0.0d)
    )
    val cw = TensorHelpers.fromVec(classWeights, device, precision)

    var i = 0
    var lastLoss = Double.MaxValue
    while (i < epochs) {
      val output = model.forward(x)
      val loss: Variable = output.nllLoss(targetT, numClasses, cw)
      logger.foreach(_.info(s"$i - ${loss.toMat.raw(0)}"))
      lastLoss = loss.toMat.raw(0)
      val gradients = model.gradients(loss)
      optim.step(gradients)
      i += 1
    }

    x.value.release
    targetT.release
    cw.release
    optim.release()
    pool.releaseAll()
    (model.asEval, lastLoss)
  }

  implicit val trainingMode: TrainingMode[Nnrf] = {
    TrainingMode.make[Nnrf](
      m => m.copy(levels = m.levels.map(_.asEval)),
      m => m.copy(levels = m.levels.map(_.asTraining))
    )
  }
  implicit def load(implicit l: Load[NnrfCell]) = Load.make[Nnrf] {
    m => parameters =>
      implicit val pool = m.levels.head.weights.pool
      val groups = parameters.grouped(3).toList
      val loadedlevels = (groups zip m.levels).map {
        case (params, cell) => cell.load(params)
      }

      m.copy(levels = loadedlevels.toIndexedSeq)
  }
  def apply(
      levels: Int,
      numFeatures: Int,
      totalDataFeatures: Int,
      out: Int,
      tOpt: TensorOptions,
      fullBatchData: Option[Variable]
  )(implicit pool: AllocatedVariablePool): Nnrf =
    Nnrf(
      0 until levels.toInt map (i =>
        NnrfCell(
          math.pow(2d, i).toInt,
          numFeatures,
          totalDataFeatures,
          tOpt,
          fullBatchData
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
    featuresPerNode: Int,
    train: Boolean,
    fullBatchData: Option[Variable]
) extends GenericModule[
      (Variable, Variable),
      (Variable)
    ] {

  override val state = List(
    features -> NnrfCell.Features,
    weights -> NnrfCell.Weights,
    biases -> NnrfCell.Bias
  )

  val subsetX: Option[Variable] = {
    fullBatchData.map(v => subsetData(v))
  }
  def subsetData(data: Variable) = {
    val fLong =
      const(ATen._cast_Long(features.value, false))(features.pool).releasable
    data
      .indexSelect(1, fLong)
      .releaseWithVariable(fLong)
      .view(List(data.shape(0).toInt, -1, featuresPerNode))
      .transpose(0, 1)
      .keep
  }

  def asEval = {
    subsetX.foreach(_.value.release)
    copy(train = false, fullBatchData = None)
  }

  def forward(tuple: (Variable, Variable)) = {
    val (x, s) = tuple
    val features = subsetX.getOrElse(subsetData(x))
    val r = (features.bmm(weights) + biases).relu
    r * s
  }

}

object NnrfCell {

  implicit val trainingMode: TrainingMode[NnrfCell] =
    TrainingMode.make[NnrfCell](
      m => m.asEval,
      m => m.copy(train = true)
    )
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
      tOpt: TensorOptions,
      fullBatchData: Option[Variable]
  )(implicit pool: AllocatedVariablePool): NnrfCell = {
    val in = numFeatures
    val out = 2
    NnrfCell(
      train = true,
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
            TensorHelpers.precision(tOpt).get
          )
      ),
      featuresPerNode = numFeatures,
      fullBatchData = fullBatchData
    )

  }

}
