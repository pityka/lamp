package lamp.tabular

import aten.Tensor
import lamp.TensorHelpers
import org.saddle._
import lamp.autograd.const
import lamp.nn._
import lamp.Device
import lamp.FloatingPointPrecision
import lamp.data.BatchStream
import lamp.autograd.Variable
import cats.effect.Resource
import cats.effect.IO
import lamp.data.IOLoops
import scribe.Logger
import lamp.data.TrainingCallback
import lamp.CPU
import cats.implicits._
import lamp.SinglePrecision
import org.saddle.index.IndexIntRange
import lamp.DoublePrecision
import lamp.CudaDevice
import lamp.RegressionTree
import _root_.lamp.ClassificationTree
import lamp.Scope
import lamp.STen
import lamp.Movable
import lamp.STenOptions

sealed trait BaseModel
case class KnnBase(
    k: Int,
    features: STen,
    predictedFeatures: Seq[STen],
    target: STen
) extends BaseModel
object KnnBase {
  implicit object IsMovable extends Movable[KnnBase] {
    def list(m: KnnBase) =
      (m.target.value +: m.features.value +: m.predictedFeatures.map(_.value)).toList
  }
}
case class ExtratreesBase(
    trees: Either[Seq[ClassificationTree], Seq[RegressionTree]]
) extends BaseModel

case class NNBase(hiddenSize: Int, state: Seq[STen]) extends BaseModel

object NNBase {
  implicit object IsMovable extends Movable[NNBase] {
    def list(m: NNBase) = m.state.toList.map(_.value)
  }
}

object BaseModel {
  implicit object IsMovable extends Movable[BaseModel] {
    def list(m: BaseModel) = m match {
      case m: KnnBase        => KnnBase.IsMovable.list(m)
      case m: NNBase         => NNBase.IsMovable.list(m)
      case _: ExtratreesBase => Nil
    }
  }
}

sealed trait Metadata
case object Numerical extends Metadata
case class Categorical(classes: Int) extends Metadata
object Categorical {
  import upickle.default.{ReadWriter => RW, macroRW}
  implicit val rw: RW[Categorical] = macroRW
}
object Metadata {
  import upickle.default.{ReadWriter => RW, macroRW}
  implicit val rw: RW[Metadata] = macroRW
  implicit val mv = Movable.empty[Metadata]
}

sealed trait TargetType
case object Regression extends TargetType
case class Classification(classes: Int, weights: Seq[Double]) extends TargetType
object Classification {
  import upickle.default.{ReadWriter => RW, macroRW}
  implicit val rw: RW[Classification] = macroRW
}
case object ECDFRegression extends TargetType

object TargetType {
  import upickle.default.{ReadWriter => RW, macroRW}
  implicit val rw: RW[TargetType] = macroRW
}

object EnsembleModel {
  def train(
      features: STen,
      target: STen,
      dataLayout: Seq[Metadata],
      targetType: TargetType,
      device: Device,
      logger: Option[Logger],
      learningRate: Double = 0.0001,
      minibatchSize: Int = 512,
      knnMinibatchSize: Int = 512,
      rng: org.saddle.spire.random.Generator
  )(implicit scope: Scope) = {
    val precision =
      if (features.options.isDouble) DoublePrecision
      else if (features.options.isFloat) SinglePrecision
      else throw new RuntimeException("Expected float or double tensor")
    val numInstances = features.sizes.apply(0).toInt

    val cvFolds =
      AutoLoop.makeCVFolds(
        numInstances,
        k = 10,
        5,
        rng
      )
    val ensembleFolds =
      AutoLoop
        .makeCVFolds(numInstances, k = 10, 5, rng)

    logger.foreach(
      _.info(
        s"Number of folds for base models: ${cvFolds.size}, for ensemble selection: ${ensembleFolds.size} "
      )
    )
    Scope.bracket { scope =>
      AutoLoop.train(
        dataFullbatch = features,
        targetFullbatch = target,
        folds = cvFolds,
        targetType = targetType,
        dataLayout = dataLayout,
        epochs = Seq(4, 8, 16, 32, 64, 128, 256),
        weighDecays = Seq(0.0001, 0.001, 0.005),
        dropouts = Seq(0.01d, 0.1, 0.5, 0.95),
        knnK = Seq(5, 25),
        extratreesK = Seq(50),
        extratreesM = Seq(50),
        extratreesNMin = Seq(2, 5),
        extratreeParallelism = 8,
        learningRate = learningRate,
        hiddenSizes = Seq(32, 128),
        device = device,
        precision = precision,
        minibatchSize = minibatchSize,
        knnMinibatchSize = knnMinibatchSize,
        logger = logger,
        ensembleFolds = ensembleFolds,
        rng = rng
      )(scope)
    }
  }

  implicit object IsMovable extends Movable[EnsembleModel] {
    def list(movable: EnsembleModel): List[Tensor] =
      (movable.selectionModels.flatMap(basemodel =>
        BaseModel.IsMovable.list(basemodel)
      ) ++ movable.baseModels.flatten.flatMap(basemodel =>
        BaseModel.IsMovable.list(basemodel)
      )).toList
  }
}

case class EnsembleModel(
    selectionModels: Seq[BaseModel],
    baseModels: Seq[Seq[BaseModel]],
    dataLayout: Seq[Metadata],
    targetType: TargetType,
    precision: FloatingPointPrecision,
    validationLosses: Seq[Double]
) {

  def predict(data: STen)(implicit scope: Scope): IO[STen] = {
    Scope.bracket(scope) { implicit scope =>
      IO {
        val device = {
          val t = selectionModels.head match {
            case NNBase(_, state)        => state.head.options
            case KnnBase(_, state, _, _) => state.options
            case _                       => STen.dOptions
          }
          if (t.isCPU) CPU
          else CudaDevice(t.deviceIndex)
        }
        val dataOnDevice = device.to(data)
        val tOpt = device.options(precision)
        val outputSize = targetType match {
          case Regression                 => 1
          case ECDFRegression             => 2
          case Classification(classes, _) => classes
        }
        val softmax = targetType match {
          case Regression => false
          case _          => true
        }
        val selectFirstOutputDimension = targetType match {
          case ECDFRegression => true
          case _              => false
        }

        val (numerical0, categorical0) =
          AutoLoop.separateFeatures(dataOnDevice, dataLayout)
        val numerical = const(numerical0)
        val categorical = categorical0.map(v => const(v._1) -> v._2)

        val basePredictions = Scope { implicit scope =>
          baseModels.map {
            case averagableModels =>
              val averagablePredictions = averagableModels.map {
                case ExtratreesBase(trees) =>
                  val features = AutoLoop.makeFusedFeatureMatrix(
                    data,
                    dataLayout,
                    Nil,
                    precision,
                    device
                  )
                  val featuresJ = features.toMat
                  val predicted = trees match {
                    case Left(trees) =>
                      lamp.extratrees
                        .predictClassification(trees, featuresJ)
                        .map(v => math.log(v + 1e-6))
                    case Right(trees) =>
                      Mat(lamp.extratrees.predictRegression(trees, featuresJ))
                  }
                  STen.fromMat(predicted, device, precision)

                case KnnBase(k, features, predictedFeatures, target) =>
                  assert(predictedFeatures.isEmpty)
                  val prediction = AutoLoop
                    .trainAndPredictKnn(
                      k = k,
                      data = features,
                      predictables = dataOnDevice,
                      predictionsOnBasemodels = Nil,
                      predictablesPredictionsOnBasemodels = Nil,
                      target = target,
                      targetType = targetType,
                      dataLayout = dataLayout,
                      device = device,
                      precision = precision,
                      minibatchSize = 100
                    )(scope)
                    .unsafeRunSync
                  prediction
                case NNBase(hiddenSize, state) =>
                  val model = AutoLoop
                    .makeModel(
                      dataLayout,
                      0d,
                      hiddenSize,
                      outputSize,
                      softmax,
                      selectFirstOutputDimension,
                      tOpt
                    )

                  model.load(state)
                  val prediction =
                    model.asEval.forward(
                      (
                        categorical
                          .map(_._1),
                        numerical
                      )
                    )
                  prediction.value.cloneTensor
              }
              val stacked = STen.stack(averagablePredictions, 0)
              val meaned = stacked.mean(List(0), false)
              meaned
          }
        }

        val numericalWithPredictions =
          const(STen.cat(numerical.value +: basePredictions.toArray, 1))

        val averagablePredictions = Scope { implicit scope =>
          selectionModels.map {
            case ExtratreesBase(trees) =>
              val features = AutoLoop.makeFusedFeatureMatrix(
                data,
                dataLayout,
                basePredictions,
                precision,
                CPU
              )
              val featuresJ = features.toMat
              val predicted = trees match {
                case Left(trees) =>
                  lamp.extratrees
                    .predictClassification(trees, featuresJ)
                    .map(v => math.log(v + 1e-6))
                case Right(trees) =>
                  Mat(lamp.extratrees.predictRegression(trees, featuresJ))
              }
              STen.fromMat(predicted, device, precision)
            case KnnBase(k, features, predictedFeatures, target) =>
              val prediction = AutoLoop
                .trainAndPredictKnn(
                  k = k,
                  data = features,
                  predictables = dataOnDevice,
                  predictionsOnBasemodels = predictedFeatures,
                  predictablesPredictionsOnBasemodels = basePredictions,
                  target = target,
                  targetType = targetType,
                  dataLayout = dataLayout,
                  device = device,
                  precision = precision,
                  minibatchSize = 100
                )(scope)
                .unsafeRunSync
              prediction
            case NNBase(hiddenSize, state) =>
              val model = AutoLoop
                .makeModel(
                  dataLayout ++
                    (0 until basePredictions
                      .map(_.sizes.apply(1))
                      .sum
                      .toInt)
                      .map(_ => Numerical),
                  0d,
                  hiddenSize,
                  outputSize,
                  softmax,
                  selectFirstOutputDimension,
                  tOpt
                )

              model.load(state)

              val prediction = model.asEval.forward(
                (
                  categorical.map(_._1),
                  numericalWithPredictions
                )
              )
              prediction.value.cloneTensor
          }
        }
        val stacked = STen.stack(averagablePredictions, 0)
        val meaned = stacked.mean(List(0), false)
        meaned
      }

    }
  }
}

object AutoLoop {

  def makeCVFolds(
      length: Int,
      k: Int,
      repeat: Int,
      rng: org.saddle.spire.random.Generator
  ) = {
    val all = IndexIntRange(length).toVec.toArray
    0 until repeat flatMap { _ =>
      val shuffled = org.saddle.array.shuffle(all, rng)
      val folds = 0 until k map (_ => org.saddle.Buffer.empty[Int]) toArray
      var i = 0
      val n = shuffled.length
      while (i < n) {
        folds(i % k).+=(shuffled(i))
        i += 1
      }
      val groups = folds.map(_.toArray)
      assert(all.toSet == groups.flatten.toSet)
      groups.map { holdout =>
        val set = holdout.toSet
        val training = all.filterNot(i => set.contains(i))
        (training.toSeq, holdout.toSeq)
      }
    }
  }

  private[lamp] def categoricalEmbeddingDimensions(classes: Int) =
    math.min(100, (1.6 * math.pow(classes.toDouble, 0.56)).toInt)

  private[lamp] def minibatchesFromFull(
      minibatchSize: Int,
      dropLast: Boolean,
      numericalFeatures: STen,
      categoricalFeatures: Seq[STen],
      target: STen,
      device: Device,
      rng: org.saddle.spire.random.Generator
  ) = {
    def makeNonEmptyBatch(idx: Array[Int]) = {
      BatchStream.scopeInResource.map { implicit scope =>
        val idxT = STen.fromLongVec(idx.toVec.map(_.toLong))
        val numCl = numericalFeatures.index(idxT)
        val catCls = categoricalFeatures.map { t =>
          val r = Scope { implicit scope =>
            val tmp = t.index(idxT)
            device.to(tmp)
          }
          const(r)
        }
        val tcl = target.index(idxT)
        val d1 = device.to(numCl)
        val d2 = device.to(tcl)
        Some(((catCls, const(d1)), d2)): Option[
          ((Seq[Variable], Variable), STen)
        ]

      }
    }
    val emptyResource =
      Resource.pure[IO, Option[((Seq[Variable], Variable), STen)]](None)

    val idx = {
      val t = array
        .shuffle(array.range(0, numericalFeatures.sizes.head.toInt), rng)
        .grouped(minibatchSize)
        .toList
      if (dropLast) t.dropRight(1)
      else t
    }

    val batchStream = new BatchStream[(Seq[Variable], Variable)] {
      private var remaining = idx
      def nextBatch: Resource[IO, Option[((Seq[Variable], Variable), STen)]] =
        remaining match {
          case Nil => emptyResource
          case x :: tail =>
            val r = makeNonEmptyBatch(x)
            remaining = tail
            r
        }
    }

    (batchStream, idx.size)

  }

  implicit val ev1 = Movable.empty[ClassificationTree]
  implicit val ev2 = Movable.empty[RegressionTree]

  private[lamp] def separateFeatures(
      data: STen,
      dataLayout: Seq[Metadata]
  )(implicit scope: Scope) = {
    val numericalIdx = dataLayout.zipWithIndex.collect {
      case (Numerical, idx) => idx
    }
    Scope { implicit scope =>
      val numericalIdxTensor =
        STen.fromLongVec(
          numericalIdx.map(_.toLong).toVec,
          TensorHelpers.device(data.value)
        )
      val numericalSubset = data.indexSelect(1, numericalIdxTensor)
      val categoricals = dataLayout.zipWithIndex.collect {
        case (Categorical(numClasses), idx) =>
          val long = Scope { implicit scope =>
            val selected = data.select(1, idx)
            selected.castToLong
          }
          (long, numClasses)
      }
      (numericalSubset, categoricals)
    }
  }

  private[lamp] def makeModel(
      dataLayout: Seq[Metadata],
      dropout: Double,
      hiddenSize: Int,
      outputSize: Int,
      softmax: Boolean,
      selectFirstOutputDimension: Boolean,
      modelTensorOptions: STenOptions
  )(implicit pool: Scope) = {
    val numericalCount = dataLayout.count {
      case Numerical => true
      case _         => false
    }
    val categoricalEmbeddingSizes = dataLayout.collect {
      case Categorical(classes) =>
        (classes, categoricalEmbeddingDimensions(classes))
    }

    val embedding = TabularEmbedding.make(
      categoricalClassesWithEmbeddingDimensions = categoricalEmbeddingSizes,
      modelTensorOptions
    )

    val residual = TabularResidual.make(
      inChannels =
        categoricalEmbeddingSizes.map(_._2).sum + numericalCount,
      hiddenChannels = hiddenSize,
      outChannels = outputSize,
      tOpt = modelTensorOptions,
      dropout = dropout
    )
    val model = sequence(
      embedding,
      residual,
      if (softmax && !selectFirstOutputDimension)
        Fun(implicit pool => _.logSoftMax(dim = 1))
      else if (softmax && selectFirstOutputDimension)
        Fun(implicit pool =>
          _.logSoftMax(dim = 1).select(1, 0).view(List(-1, 1))
        )
      else Fun(_ => identity)
    )

    model

  }

  private[lamp] def trainAndPredictKnn(
      k: Int,
      data: STen,
      predictables: STen,
      predictionsOnBasemodels: Seq[STen],
      predictablesPredictionsOnBasemodels: Seq[STen],
      target: STen,
      targetType: TargetType,
      dataLayout: Seq[Metadata],
      device: Device,
      precision: FloatingPointPrecision,
      minibatchSize: Int
  )(outerScope: Scope) =
    Scope.bracket(outerScope) { implicit scope =>
      IO {

        val trainingFeatures =
          makeFusedFeatureMatrix(
            data,
            dataLayout,
            predictionsOnBasemodels,
            precision,
            device
          )

        val predictableFeatures =
          makeFusedFeatureMatrix(
            predictables,
            dataLayout,
            predictablesPredictionsOnBasemodels,
            precision,
            device
          )

        val predicteds = {
          val indices = lamp.knn.knnMinibatched(
            trainingFeatures,
            predictableFeatures,
            k,
            lamp.knn.SquaredEuclideanDistance,
            minibatchSize
          )
          val indicesJvm = indices.toLongMat.map(_.toInt)
          val prediction = targetType match {
            case Classification(classes, _) =>
              val targetVec =
                target.toLongMat.toVec.map(_.toInt)
              lamp.knn.classification(
                targetVec,
                indicesJvm,
                classes,
                log = true
              )
            case Regression | ECDFRegression =>
              val targetVec = target.toMat.toVec
              Mat(lamp.knn.regression(targetVec, indicesJvm))
          }

          STen.fromMat(prediction, CPU, precision),

        }

        predicteds
      }

    }

  private[lamp] def makeFusedFeatureMatrix(
      data: STen,
      dataLayout: Seq[Metadata],
      predictionsOnBasemodels: Seq[STen],
      precision: FloatingPointPrecision,
      device: Device
  )(implicit scope: Scope) = {
    Scope { implicit scope =>
      val (trainNumerical, trainCategorical) = Scope { implicit scope =>
        val (num, cat) = separateFeatures(data, dataLayout)
        if (predictionsOnBasemodels.isEmpty) (num, cat)
        else {
          val numWithPredictions =
            STen.cat(num +: predictionsOnBasemodels.toArray, 1)
          (numWithPredictions, cat)
        }
      }
      val categoricalOneHot = trainCategorical.map {
        case (t, numClasses) =>
          Scope { implicit scope =>
            val t2 = t.oneHot(numClasses)
            precision match {
              case SinglePrecision => t2.castToFloat
              case DoublePrecision => t2.castToDouble
            }

          }
      }
      device.to(STen.cat((categoricalOneHot :+ trainNumerical), 1))

    }

  }
  private[lamp] def trainAndPredictExtratrees(
      k: Int,
      m: Int,
      nMin: Int,
      parallelism: Int,
      data: STen,
      predictables: STen,
      predictionsOnBasemodels: Seq[STen],
      predictablesPredictionsOnBasemodels: Seq[STen],
      target: STen,
      targetType: TargetType,
      dataLayout: Seq[Metadata]
  )(outerScope: Scope) = {

    Scope.bracket(outerScope) { implicit scope =>
      IO {
        val precision = TensorHelpers.precision(data.value).get
        val trainingFeatures = makeFusedFeatureMatrix(
          data,
          dataLayout,
          predictionsOnBasemodels,
          precision,
          CPU
        )
        val trainingFeaturesJvm = trainingFeatures.toMat

        val predictableFeatures = makeFusedFeatureMatrix(
          predictables,
          dataLayout,
          predictablesPredictionsOnBasemodels,
          precision,
          CPU
        )
        val predictableFeaturesJvm = predictableFeatures.toMat

        targetType match {
          case Classification(classes, _) =>
            val targetVec =
              target.toLongMat.toVec.map(_.toInt)
            val trained = lamp.extratrees.buildForestClassification(
              data = trainingFeaturesJvm,
              target = targetVec,
              sampleWeights = None,
              numClasses = classes,
              nMin = nMin,
              k = k,
              m = m,
              parallelism = parallelism
            )
            val prediction = lamp.extratrees
              .predictClassification(trained, predictableFeaturesJvm)
              .map(v => math.log(v + 1e-6))
            val predictionT = STen.fromMat(prediction, CPU, precision)
            (predictionT, Left(trained))

          case Regression | ECDFRegression =>
            val targetVec = target.toMat.toVec
            val trained = lamp.extratrees.buildForestRegression(
              data = trainingFeaturesJvm,
              target = targetVec,
              nMin = nMin,
              k = k,
              m = m,
              parallelism = parallelism
            )
            val prediction = Mat(
              lamp.extratrees
                .predictRegression(trained, predictableFeaturesJvm)
            )
            val predictionT = STen.fromMat(prediction, CPU, precision)
            (predictionT, Right(trained))
        }

      }
    }
  }

  private[lamp] def trainAndPredict1(
      data: STen,
      predictables: STen,
      predictionsOnBasemodels: Seq[STen],
      predictablesPredictionsOnBasemodels: Seq[STen],
      target: STen,
      targetType: TargetType,
      dataLayout: Seq[Metadata],
      epochs: Seq[Int],
      learningRate: Double,
      weightDecay: Double,
      dropout: Double,
      hiddenSize: Int,
      device: Device,
      precision: FloatingPointPrecision,
      minibatchSize: Int,
      logger: Option[Logger],
      rng: org.saddle.spire.random.Generator
  )(implicit scope: Scope) =
    Scope.bracket { implicit scope =>
      val modelTensorOptions = device.options(precision)
      val outputSize = targetType match {
        case Regression                 => 1
        case ECDFRegression             => 2
        case Classification(classes, _) => classes
      }
      val softmax = targetType match {
        case Regression => false
        case _          => true
      }
      val selectFirstOutputDimension = targetType match {
        case ECDFRegression => true
        case _              => false
      }
      val model =
        makeModel(
          dataLayout ++
            (0 until predictionsOnBasemodels
              .map(_.sizes.apply(1))
              .sum
              .toInt)
              .map(_ => Numerical),
          dropout,
          hiddenSize,
          outputSize,
          softmax,
          selectFirstOutputDimension,
          modelTensorOptions
        )
      logger.foreach(
        _.info(s"Learnable parameters: " + model.learnableParameters)
      )
      val lossFunction = targetType match {
        case Regression     => (LossFunctions.L1Loss)
        case ECDFRegression => (LossFunctions.L1Loss)
        case Classification(classes, classWeights) =>
          val classWeightsT =
            STen.fromVec(classWeights.toVec, device, precision)
          (
            LossFunctions.NLL(
              numClasses = classes,
              classWeights = classWeightsT
            ),
          )
      }

      val (trainNumerical, trainCategorical) = Scope { implicit scope =>
        val (num, cat) = separateFeatures(data, dataLayout)
        if (predictionsOnBasemodels.isEmpty)
          (num, cat.map(_._1))
        else {
          val numWithPredictions =
            STen.cat(num +: predictionsOnBasemodels, 1)
          (numWithPredictions, cat.map(_._1))
        }
      }

      def miniBatches =
        minibatchesFromFull(
          minibatchSize = minibatchSize,
          dropLast = false,
          numericalFeatures = trainNumerical,
          categoricalFeatures = trainCategorical,
          target = target,
          device = device,
          rng
        )

      val maxEpochs = epochs.max

      val modelWithOptimizer =
        SupervisedModel(model, lossFunction).zipOptimizer(
          AdamW.factory(
            weightDecay = simple(weightDecay),
            learningRate = simple(learningRate)
          )
        )

      val learningRateSchedule = LearningRateSchedule.stepAfter(
        steps = (maxEpochs * 0.8).toLong,
        factor = 0.1
      )

      def batchStream = miniBatches._1

      val (predictNumerical, predictCategorical) = Scope { implicit scope =>
        val (num, cat) = separateFeatures(predictables, dataLayout)
        if (predictablesPredictionsOnBasemodels.isEmpty) (num, cat.map(_._1))
        else {
          val numWithPredictions =
            STen.cat(num +: predictablesPredictionsOnBasemodels, 1)
          (numWithPredictions, cat.map(_._1))
        }
      }

      def loop(
          epoch: Int,
          predictedAtEpochs: Seq[(Int, STen, Seq[(STen, PTag)])]
      ): IO[Seq[(Int, STen, Seq[(STen, PTag)])]] =
        if (epoch > maxEpochs) {
          IO {
            logger.foreach(_.info(s"Max epochs ($maxEpochs) reached."))
            predictedAtEpochs
          }
        } else {
          for {
            _ <- IOLoops.oneEpoch(
              epoch,
              TrainingCallback.noop,
              modelWithOptimizer,
              batchStream,
              logger,
              learningRateSchedule.factor(epoch.toLong, None),
              None
            )
            next <- loop(
              epoch + 1, {
                logger.foreach(_.info(s"Epoch $epoch done."))
                if (epochs.contains(epoch)) {
                  Scope {
                    implicit scope =>
                      val predicted = modelWithOptimizer.model.module
                        .forward(
                          (
                            predictCategorical.map(v => const(device.to(v))),
                            const(device.to(predictNumerical))
                          )
                        )
                      val copy =
                        predicted.value.copyTo(predictNumerical.options)
                      val modelState = model.state.map {
                        case (stateVar, ptag) =>
                          Scope { implicit scope =>
                            val clone = stateVar.value.cloneTensor
                            val onCpu = clone.copyToDevice(CPU)
                            (onCpu, ptag)
                          }
                      }

                      (epoch, copy, modelState) +: predictedAtEpochs
                  }
                } else predictedAtEpochs
              }
            )
          } yield next
        }

      val predicteds = loop(0, Nil)

      predicteds

    }

  private[lamp] def aggregatePredictionsAndModelsPerEpoch[ModelState](
      byEpoch: Seq[(Int, Seq[(Int, STen, ModelState, Seq[Int])])],
      expectedRows: Long
  )(implicit scope: Scope): Seq[(Int, STen, Seq[ModelState])] =
    byEpoch.map {
      case (epoch, folds) =>
        val concat = Scope { implicit scope =>
          val predictionsWithInstanceIDs: Vector[(STen, Seq[Int])] =
            folds.map {
              case (_, prediction, _, idx) => (prediction, idx)
            }.toVector
          // (instanceID, index within fold, index of fold)
          val indices: Seq[(Int, Int, Int)] =
            predictionsWithInstanceIDs.zipWithIndex.flatMap {
              case ((_, instanceIdx), foldIdx) =>
                instanceIdx.zipWithIndex.map {
                  case (instanceIdx, innerIdx) =>
                    (instanceIdx, innerIdx, foldIdx)
                }
            }

          val averagedPredictionsInEpoch = indices
            .groupBy(_._1)
            .toSeq
            .map {
              case (instanceIdx, group) =>
                Scope { implicit scope =>
                  val locations = group.map {
                    case (_, innerIdx, foldIdx) => (innerIdx, foldIdx)
                  }
                  val tensorsOfPredictionsOfInstance = locations.map {
                    case (innerIdx, foldIdx) =>
                      predictionsWithInstanceIDs(foldIdx)._1.select(
                        0,
                        innerIdx
                      )

                  }
                  val concat =
                    STen.stack(
                      tensorsOfPredictionsOfInstance,
                      0
                    )
                  val mean =
                    concat.mean(Array(0), false)
                  (instanceIdx, mean)
                }
            }
            .sortBy(_._1)
            .map(_._2)

          val concat =
            STen.stack(averagedPredictionsInEpoch, 0)
          assert(
            concat.sizes.head == expectedRows,
            s"${concat.sizes.head} != $expectedRows"
          )
          assert(concat.sizes.size == 2)
          concat
        }
        (epoch, concat, folds.map(_._3))

    }

  private[lamp] def slice(
      dataFullbatch: STen,
      targetFullbatch: STen,
      predictions: Seq[STen],
      trainIdx: Seq[Int],
      predictIdx: Seq[Int]
  )(implicit scope: Scope) = Scope { implicit scope =>
    val trainIdxT =
      STen.fromLongVec(
        trainIdx.toVec.map(_.toLong),
        CPU
      )
    val predictIdxT =
      STen.fromLongVec(
        predictIdx.toVec.map(_.toLong),
        CPU
      )
    val trainFeatures =
      dataFullbatch.indexSelect(0, trainIdxT)
    val trainTarget =
      targetFullbatch.indexSelect(0, trainIdxT)
    val predictableFeatures =
      dataFullbatch.indexSelect(0, predictIdxT)

    val trainPredictions =
      predictions.map { t => t.indexSelect(0, trainIdxT) }
    val predictablePredictions =
      predictions.map { t => t.indexSelect(0, predictIdxT) }

    (
      trainFeatures,
      trainPredictions,
      trainTarget,
      predictableFeatures,
      predictablePredictions
    )
  }

  def trainAndAverageFoldsExtratrees(
      k: Int,
      m: Int,
      nMin: Int,
      parallelism: Int,
      dataFullbatch: STen,
      targetFullbatch: STen,
      predictions: Seq[STen],
      folds: Seq[(Seq[Int], Seq[Int])],
      targetType: TargetType,
      dataLayout: Seq[Metadata],
      logger: Option[Logger]
  )(outerScope: Scope) = {
    def trainedFolds(scope: Scope) =
      folds.zipWithIndex
        .map {
          case ((trainIdx, predictIdx), foldIdx) =>
            assert((trainIdx.toSet & predictIdx.toSet).size == 0)
            Scope.bracket(scope) { implicit scope =>
              for {
                sliced <- IO {
                  slice(
                    dataFullbatch = dataFullbatch,
                    targetFullbatch = targetFullbatch,
                    predictions = predictions,
                    trainIdx = trainIdx,
                    predictIdx = predictIdx
                  )
                }
                (
                  trainFeatures,
                  trainPredictions,
                  trainTarget,
                  predictableFeatures,
                  predictablePredictions
                ) = sliced
                _ <- IO {
                  logger.foreach(
                    _.info(
                      s"Train KNN model, fold ${foldIdx + 1} / ${folds.size}"
                    )
                  )
                }

                result <- trainAndPredictExtratrees(
                  k = k,
                  m = m,
                  nMin = nMin,
                  parallelism = parallelism,
                  data = trainFeatures,
                  predictables = predictableFeatures,
                  predictionsOnBasemodels = trainPredictions,
                  predictablesPredictionsOnBasemodels = predictablePredictions,
                  target = trainTarget,
                  targetType = targetType,
                  dataLayout = dataLayout
                )(scope).map {
                  case (prediction, model) =>
                    (
                      0,
                      prediction,
                      model,
                      predictIdx
                    )
                }
              } yield {

                result
              }
            }
        }
        .toList
        .sequence

    Scope.bracket(outerScope) { scope =>
      for {
        trainedFolds <- trainedFolds(scope)
        aggregatedByEpochs <- IO {
          logger.foreach(
            _.info(
              s"Folds done. Aggregating predictions.."
            )
          )
          val byEpoch = trainedFolds.groupBy {
            case (epoch, _, _, _) => epoch
          }.toSeq

          aggregatePredictionsAndModelsPerEpoch(
            byEpoch,
            dataFullbatch.sizes.head
          )(scope).map {
            case (epoch, prediction, models) =>
              (epoch, prediction, models.map {
                case trees =>
                  ExtratreesBase(trees)
              })
          }
        }
      } yield aggregatedByEpochs
    }
  }
  def trainAndAverageFoldsKnn(
      k: Int,
      dataFullbatch: STen,
      targetFullbatch: STen,
      predictions: Seq[STen],
      folds: Seq[(Seq[Int], Seq[Int])],
      targetType: TargetType,
      dataLayout: Seq[Metadata],
      device: Device,
      precision: FloatingPointPrecision,
      minibatchSize: Int,
      logger: Option[Logger]
  )(outerScope: Scope) = {
    def trainedFolds(scope: Scope) =
      folds.zipWithIndex
        .map {
          case ((trainIdx, predictIdx), foldIdx) =>
            assert((trainIdx.toSet & predictIdx.toSet).size == 0)
            Scope.bracket(scope) { scope =>
              for {
                sliced <- IO {
                  slice(
                    dataFullbatch = dataFullbatch,
                    targetFullbatch = targetFullbatch,
                    predictions = predictions,
                    trainIdx = trainIdx,
                    predictIdx = predictIdx
                  )(scope)
                }
                (
                  trainFeatures,
                  trainPredictions,
                  trainTarget,
                  predictableFeatures,
                  predictablePredictions
                ) = sliced
                _ <- IO {
                  logger.foreach(
                    _.info(
                      s"Train KNN model, fold ${foldIdx + 1} / ${folds.size}"
                    )
                  )
                }

                result <- trainAndPredictKnn(
                  k = k,
                  data = trainFeatures,
                  predictables = predictableFeatures,
                  predictionsOnBasemodels = trainPredictions,
                  predictablesPredictionsOnBasemodels = predictablePredictions,
                  target = trainTarget,
                  targetType = targetType,
                  dataLayout = dataLayout,
                  device = device,
                  precision = precision,
                  minibatchSize = minibatchSize
                )(scope).map { prediction =>
                  (
                    0,
                    prediction,
                    (k, trainFeatures, trainPredictions, trainTarget),
                    predictIdx
                  )
                }
              } yield {
                result
              }
            }
        }
        .toList
        .sequence

    Scope.bracket(outerScope) { scope =>
      for {
        trainedFolds <- trainedFolds(scope)
        aggregatedByEpochs <- IO {
          logger.foreach(
            _.info(
              s"Folds done. Aggregating predictions.."
            )
          )
          val byEpoch = trainedFolds.groupBy {
            case (epoch, _, _, _) => epoch
          }.toSeq

          aggregatePredictionsAndModelsPerEpoch(
            byEpoch,
            dataFullbatch.sizes.head
          )(scope).map {
            case (epoch, prediction, models) =>
              (epoch, prediction, models.map {
                case (k, features, predictedFeatures, target) =>
                  KnnBase(k, features, predictedFeatures, target)
              })
          }
        }
      } yield aggregatedByEpochs
    }
  }
  def trainAndAverageFolds(
      dataFullbatch: STen,
      targetFullbatch: STen,
      predictions: Seq[STen],
      folds: Seq[(Seq[Int], Seq[Int])],
      targetType: TargetType,
      dataLayout: Seq[Metadata],
      epochs: Seq[Int],
      weightDecay: Double,
      dropout: Double,
      hiddenSize: Int,
      learningRate: Double,
      device: Device,
      precision: FloatingPointPrecision,
      minibatchSize: Int,
      logger: Option[Logger],
      rng: org.saddle.spire.random.Generator
  )(outerScope: Scope) = {
    def trainedFolds(scope: Scope) =
      folds.zipWithIndex
        .map {
          case ((trainIdx, predictIdx), foldIdx) =>
            assert((trainIdx.toSet & predictIdx.toSet).size == 0)
            Scope.bracket(scope) { implicit scope =>
              for {
                sliced <- IO {
                  slice(
                    dataFullbatch = dataFullbatch,
                    targetFullbatch = targetFullbatch,
                    predictions = predictions,
                    trainIdx = trainIdx,
                    predictIdx = predictIdx
                  )
                }
                (
                  trainFeatures,
                  trainPredictions,
                  trainTarget,
                  predictableFeatures,
                  predictablePredictions
                ) = sliced
                _ <- IO {
                  logger.foreach(
                    _.info(
                      s"Train model, hd=$hiddenSize wd=$weightDecay, dro=$dropout fold ${foldIdx + 1} / ${folds.size}"
                    )
                  )
                }

                result <- trainAndPredict1(
                  data = trainFeatures,
                  predictables = predictableFeatures,
                  predictionsOnBasemodels = trainPredictions,
                  predictablesPredictionsOnBasemodels = predictablePredictions,
                  target = trainTarget,
                  targetType = targetType,
                  dataLayout = dataLayout,
                  epochs = epochs,
                  learningRate = learningRate,
                  weightDecay = weightDecay,
                  dropout = dropout,
                  hiddenSize = hiddenSize,
                  device = device,
                  precision = precision,
                  minibatchSize = minibatchSize,
                  logger = logger,
                  rng = rng
                ).map(_.map(v => (v._1, v._2, v._3, predictIdx)))
              } yield result

            }
        }
        .toList
        .sequence

    Scope.bracket(outerScope) { implicit scope =>
      for {
        trainedFolds <- trainedFolds(scope)
        aggregatedByEpochs <- IO {
          logger.foreach(
            _.info(
              s"Folds done. Aggregating predictions.."
            )
          )
          val byEpoch = trainedFolds.flatten.groupBy {
            case (epoch, _, _, _) => epoch
          }.toSeq

          aggregatePredictionsAndModelsPerEpoch(
            byEpoch,
            dataFullbatch.sizes.head
          ).map {
            case (epoch, prediction, models) =>
              (epoch, prediction, models.map {
                case state => NNBase(hiddenSize, state.map(_._1))
              })
          }
        }
      } yield aggregatedByEpochs
    }

  }

  def computeValidationErrors(
      pred: STen,
      targetType: TargetType,
      precision: FloatingPointPrecision,
      targetFullbatch: STen
  )(implicit scope: Scope) = Scope { implicit scope =>
    val lossFunction = targetType match {
      case Regression     => (LossFunctions.L1Loss)
      case ECDFRegression => (LossFunctions.L1Loss)
      case Classification(classes, classWeights) =>
        val classWeightsT =
          STen.fromVec(
            classWeights.toVec,
            CPU,
            precision
          )

        LossFunctions.NLL(
          numClasses = classes,
          classWeights = classWeightsT
        )

    }
    val (loss, _) =
      lossFunction.apply(
        const(pred),
        targetFullbatch
      )
    val lossM = loss.toMat.raw(0)
    lossM
  }

  def train(
      dataFullbatch: STen,
      targetFullbatch: STen,
      folds: Seq[(Seq[Int], Seq[Int])],
      targetType: TargetType,
      dataLayout: Seq[Metadata],
      epochs: Seq[Int],
      weighDecays: Seq[Double],
      dropouts: Seq[Double],
      hiddenSizes: Seq[Int],
      knnK: Seq[Int],
      knnMinibatchSize: Int,
      extratreesK: Seq[Int],
      extratreesM: Seq[Int],
      extratreesNMin: Seq[Int],
      extratreeParallelism: Int,
      learningRate: Double,
      device: Device,
      precision: FloatingPointPrecision,
      minibatchSize: Int,
      logger: Option[Logger],
      ensembleFolds: Seq[(Seq[Int], Seq[Int])],
      rng: org.saddle.spire.random.Generator
  )(outerScope: Scope) = {
    val hyperparameters =
      hiddenSizes
        .flatMap(hd =>
          weighDecays.flatMap(wd => dropouts.map(dro => (wd, dro, hd)))
        )
        .distinct

    val knnHyperparameters = knnK

    val extratreesHyperparameters = (extratreesK.flatMap { k =>
      extratreesM.flatMap { m => extratreesNMin.map { nMin => (k, m, nMin) } }
    }).distinct

    def baseModels(hp: Seq[(Double, Double, Int)])(scope: Scope) = {
      val nn = hp
        .map {
          case (wd, dro, hiddenSize) =>
            val predictionsInFolds = trainAndAverageFolds(
              dataFullbatch = dataFullbatch,
              targetFullbatch = targetFullbatch,
              predictions = Nil,
              folds = folds,
              targetType = targetType,
              dataLayout = dataLayout,
              epochs = epochs,
              weightDecay = wd,
              dropout = dro,
              hiddenSize = hiddenSize,
              learningRate = learningRate,
              device = device,
              precision = precision,
              minibatchSize = minibatchSize,
              logger = logger,
              rng = rng
            )(scope)
            predictionsInFolds.map(_.map {
              case (epoch, prediction, models) =>
                assert(prediction.isCPU)
                assert(models.forall(_.state.forall(_.isCPU)))
                (epoch, prediction, models)
            })
        }
        .toList
        .sequence
        .map(_.flatten)

      val knn = knnHyperparameters
        .map { k =>
          val predictionsInFolds = trainAndAverageFoldsKnn(
            k = k,
            dataFullbatch = dataFullbatch,
            targetFullbatch = targetFullbatch,
            predictions = Nil,
            folds = folds,
            targetType = targetType,
            dataLayout = dataLayout,
            device = device,
            precision = precision,
            minibatchSize = knnMinibatchSize,
            logger = logger
          )(scope)
          predictionsInFolds.map(_.map {
            case (epoch, prediction, models) =>
              assert(prediction.isCPU)
              assert(models.forall(_.features.isCPU))
              assert(models.forall(_.target.isCPU))
              (epoch, prediction, models)
          })
        }
        .toList
        .sequence
        .map(_.flatten)

      val extratrees = extratreesHyperparameters
        .map {
          case (k, m, nMin) =>
            val predictionsInFolds = trainAndAverageFoldsExtratrees(
              k = k,
              m = m,
              nMin = nMin,
              parallelism = extratreeParallelism,
              dataFullbatch = dataFullbatch,
              targetFullbatch = targetFullbatch,
              predictions = Nil,
              folds = folds,
              targetType = targetType,
              dataLayout = dataLayout,
              logger = logger
            )(scope)
            predictionsInFolds.map(_.map {
              case (epoch, prediction, models) =>
                assert(prediction.isCPU)
                (epoch, prediction, models)
            })
        }
        .toList
        .sequence
        .map(_.flatten)

      for {
        extratrees <- extratrees
        nn <- nn
        knn <- knn
      } yield extratrees ++ nn ++ knn

    }

    def highLevelModels(predictions: Seq[STen])(scope: Scope) = {
      val nn = hyperparameters
        .map {
          case (wd, dro, hiddenSize) =>
            for {
              trainedEnsembleFolds <- trainAndAverageFolds(
                dataFullbatch = dataFullbatch,
                targetFullbatch = targetFullbatch,
                predictions = predictions,
                folds = ensembleFolds,
                targetType = targetType,
                dataLayout = dataLayout,
                epochs = epochs,
                weightDecay = wd,
                dropout = dro,
                hiddenSize = hiddenSize,
                learningRate = learningRate,
                device = device,
                precision = precision,
                minibatchSize = minibatchSize,
                logger = logger,
                rng = rng
              )(scope)

              withValidationErrors <- IO {
                trainedEnsembleFolds.map {
                  case (_, pred, models) =>
                    assert(pred.isCPU)
                    assert(models.forall(_.state.forall(_.isCPU)))
                    val lossM = computeValidationErrors(
                      pred,
                      targetType,
                      precision,
                      targetFullbatch
                    )(scope)
                    (lossM, models)

                }
              }

            } yield withValidationErrors
        }
        .toList
        .sequence
        .map(_.flatten)

      val knn = knnHyperparameters
        .map { k =>
          for {
            trainedEnsembleFolds <- trainAndAverageFoldsKnn(
              k = k,
              dataFullbatch = dataFullbatch,
              targetFullbatch = targetFullbatch,
              predictions = predictions,
              folds = ensembleFolds,
              targetType = targetType,
              dataLayout = dataLayout,
              device = device,
              precision = precision,
              minibatchSize = knnMinibatchSize,
              logger = logger
            )(scope)

            withValidationErrors <- IO {
              trainedEnsembleFolds.map {
                case (_, pred, models) =>
                  assert(pred.isCPU)
                  assert(models.forall(_.features.isCPU))
                  assert(models.forall(_.target.isCPU))
                  val lossM = computeValidationErrors(
                    pred,
                    targetType,
                    precision,
                    targetFullbatch
                  )(scope)
                  (lossM, models)

              }
            }

          } yield withValidationErrors
        }
        .toList
        .sequence
        .map(_.flatten)

      val extratrees = extratreesHyperparameters
        .map {
          case (k, m, nMin) =>
            for {
              trainedEnsembleFolds <- trainAndAverageFoldsExtratrees(
                k = k,
                m = m,
                nMin = nMin,
                parallelism = extratreeParallelism,
                dataFullbatch = dataFullbatch,
                targetFullbatch = targetFullbatch,
                predictions = predictions,
                folds = ensembleFolds,
                targetType = targetType,
                dataLayout = dataLayout,
                logger = logger
              )(scope)

              withValidationErrors <- IO {
                trainedEnsembleFolds.map {
                  case (_, pred, models) =>
                    assert(pred.isCPU)
                    val lossM = computeValidationErrors(
                      pred,
                      targetType,
                      precision,
                      targetFullbatch
                    )(scope)
                    (lossM, models)

                }
              }

            } yield withValidationErrors
        }
        .toList
        .sequence
        .map(_.flatten)

      for {
        xt <- extratrees
        nn <- nn
        knn <- knn
      } yield xt ++ knn ++ nn

    }

    Scope.bracket(outerScope) { scope =>
      for {

        pred <- baseModels(hyperparameters)(scope = scope)
        _ <- IO {
          logger.foreach(_.info(s"${pred.size} base models done."))
        }
        trainedModelsWithValidationLosses <- highLevelModels(pred.map(_._2))(
          scope
        )

      } yield {
        logger.foreach(_.info("Training done."))
        val (selected, rejected) = {
          val sorted = trainedModelsWithValidationLosses.sortBy(_._1)
          val top = sorted.head._1
          assert(top >= 0d)
          val selected = sorted.takeWhile(_._1 <= top * 1.1)
          val rejected = sorted.dropWhile(_._1 <= top * 1.1)
          (selected, rejected)
        }
        val validationLosses = selected.map(_._1)
        logger.foreach(
          _.info(
            s"Selected models with losses: [${validationLosses
              .mkString(", ")}]. ____ Rejected: ${rejected.map(_._1)}"
          )
        )

        EnsembleModel(
          selectionModels = selected.flatMap {
            case (_, models) =>
              models
          },
          baseModels = pred.map {
            case (_, _, models) =>
              models
          },
          dataLayout = dataLayout,
          targetType = targetType,
          precision = precision,
          validationLosses = validationLosses
        )
      }
    }
  }

}
