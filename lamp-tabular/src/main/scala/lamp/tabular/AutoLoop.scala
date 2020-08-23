package lamp.tabular

import aten.Tensor
import aten.ATen
import lamp.autograd.TensorHelpers
import org.saddle._
import aten.TensorOptions
import lamp.autograd.AllocatedVariablePool
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

sealed trait BaseModel
case class KnnBase(
    k: Int,
    features: Tensor,
    predictedFeatures: Seq[Tensor],
    target: Tensor
) extends BaseModel
case class ExtratreesBase(
    trees: Either[Seq[ClassificationTree], Seq[RegressionTree]]
) extends BaseModel
case class NNBase(hiddenSize: Int, state: Seq[Tensor]) extends BaseModel

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
      features: Tensor,
      target: Tensor,
      dataLayout: Seq[Metadata],
      targetType: TargetType,
      device: Device,
      logger: Option[Logger],
      logFrequency: Int,
      learningRate: Double = 0.0001,
      minibatchSize: Int = 512,
      knnMinibatchSize: Int = 512
  ) = {
    implicit val pool = new AllocatedVariablePool
    val precision =
      if (features.options.isDouble) DoublePrecision
      else if (features.options.isFloat) SinglePrecision
      else throw new RuntimeException("Expected float or double tensor")
    val numInstances = features.sizes.apply(0).toInt

    val cvFolds =
      AutoLoop.makeCVFolds(
        numInstances,
        k = 10,
        5
      )
    val ensembleFolds =
      AutoLoop
        .makeCVFolds(numInstances, k = 10, 5)

    logger.foreach(
      _.info(
        s"Number of folds for base models: ${cvFolds.size}, for ensemble selection: ${ensembleFolds.size} "
      )
    )
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
      logFrequency = logFrequency,
      logger = logger,
      ensembleFolds = ensembleFolds,
      prescreenHyperparameters = true
    )
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

  def predict(data: Tensor)(
      implicit pool: AllocatedVariablePool
  ): Resource[IO, Tensor] = {
    Resource.make {
      IO {
        val device = {
          val t = selectionModels.head match {
            case NNBase(_, state)        => state.head.options
            case KnnBase(_, state, _, _) => state.options
            case _                       => TensorOptions.d
          }
          if (t.isCPU) CPU
          else CudaDevice(t.deviceIndex())
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

        val (numerical, categorical) =
          AutoLoop.separateFeatures(dataOnDevice, dataLayout)

        val basePredictions = baseModels.map {
          case averagableModels =>
            val averagablePredictions = averagableModels.map {
              case ExtratreesBase(trees) =>
                import lamp.syntax
                val features = AutoLoop.makeFusedFeatureMatrix(
                  data,
                  dataLayout,
                  Nil,
                  precision,
                  device
                )
                val featuresJ = features.toMat
                features.release
                val predicted = trees match {
                  case Left(trees) =>
                    lamp.extratrees
                      .predictClassification(trees, featuresJ)
                      .map(v => math.log(v + 1e-6))
                  case Right(trees) =>
                    Mat(lamp.extratrees.predictRegression(trees, featuresJ))
                }
                TensorHelpers.fromMat(predicted, device, precision)

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
                  )
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
                  .load(state)
                val prediction =
                  model.asEval.forward(
                    (
                      categorical
                        .map { case (tensor, _) => tensor }
                        .map(const(_)),
                      const(numerical)
                    )
                  )
                val copy = ATen.clone(prediction.value)
                prediction.releaseAll
                copy
            }
            val stacked = ATen.stack(averagablePredictions.toArray, 0)
            val meaned = ATen.mean_1(stacked, Array(0), false)
            averagablePredictions.foreach(_.release)
            stacked.release
            meaned
        }

        val numericalWithPredictions =
          ATen.cat(numerical +: basePredictions.toArray, 1)
        numerical.release

        val averagablePredictions = selectionModels.map {
          case ExtratreesBase(trees) =>
            import lamp.syntax
            val features = AutoLoop.makeFusedFeatureMatrix(
              data,
              dataLayout,
              basePredictions,
              precision,
              CPU
            )
            val featuresJ = features.toMat
            features.release
            val predicted = trees match {
              case Left(trees) =>
                lamp.extratrees
                  .predictClassification(trees, featuresJ)
                  .map(v => math.log(v + 1e-6))
              case Right(trees) =>
                Mat(lamp.extratrees.predictRegression(trees, featuresJ))
            }
            TensorHelpers.fromMat(predicted, device, precision)
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
              )
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
              .load(state)

            val prediction = model.asEval.forward(
              (
                categorical.map(_._1).map(const(_)),
                const(numericalWithPredictions)
              )
            )
            val copy = ATen.clone(prediction.value)
            prediction.releaseAll
            copy
        }
        val stacked = ATen.stack(averagablePredictions.toArray, 0)
        val meaned = ATen.mean_1(stacked, Array(0), false)
        averagablePredictions.foreach(_.release)
        stacked.release
        categorical.map(_._1).foreach(_.release)
        dataOnDevice.release
        numericalWithPredictions.release
        meaned
      }
    }(t => IO(t.release))
  }
}

object AutoLoop {

  private[lamp] def makeCVFolds(length: Int, k: Int, repeat: Int) = {
    val all = IndexIntRange(length).toVec.toArray
    0 until repeat flatMap { _ =>
      val shuffled = org.saddle.array.shuffle(all)
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
      numericalFeatures: Tensor,
      categoricalFeatures: Seq[Tensor],
      target: Tensor,
      device: Device
  )(implicit pool: AllocatedVariablePool) = {
    def makeNonEmptyBatch(idx: Array[Int]) = {
      Resource.make(IO {
        val idxT = TensorHelpers.fromLongVec(idx.toVec.map(_.toLong))
        val numCl = ATen.index(numericalFeatures, Array(idxT))
        val catCls = categoricalFeatures.map { t =>
          val tmp = ATen.index(t, Array(idxT))
          val r = device.to(tmp)
          tmp.release
          const(r).releasable
        }
        val tcl = ATen.index(target, Array(idxT))
        val d1 = device.to(numCl)
        val d2 = device.to(tcl)
        numCl.release
        tcl.release
        idxT.release
        Some(((catCls, const(d1).releasable), d2)): Option[
          ((Seq[Variable], Variable), Tensor)
        ]
      }) {
        case None => IO.unit
        case Some((_, b)) =>
          IO {
            b.release
          }
      }
    }
    val emptyResource =
      Resource.pure[IO, Option[((Seq[Variable], Variable), Tensor)]](None)

    val idx = {
      val t = array
        .shuffle(array.range(0, numericalFeatures.sizes.head.toInt))
        .grouped(minibatchSize)
        .toList
      if (dropLast) t.dropRight(1)
      else t
    }

    val batchStream = new BatchStream[(Seq[Variable], Variable)] {
      private var remaining = idx
      def nextBatch: Resource[IO, Option[((Seq[Variable], Variable), Tensor)]] =
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

  private[lamp] def separateFeatures(
      data: Tensor,
      dataLayout: Seq[Metadata]
  ) = {
    val numericalIdx = dataLayout.zipWithIndex.collect {
      case (Numerical, idx) => idx
    }
    val numericalIdxTensor =
      TensorHelpers.fromLongVec(
        numericalIdx.map(_.toLong).toVec,
        TensorHelpers.device(data)
      )
    val numericalSubset = ATen.index_select(data, 1, numericalIdxTensor)
    numericalIdxTensor.release
    val categoricals = dataLayout.zipWithIndex.collect {
      case (Categorical(numClasses), idx) =>
        val selected = ATen.select(data, 1, idx)
        val long = ATen._cast_Long(selected, false)
        selected.release
        (long, numClasses)
    }
    (numericalSubset, categoricals)
  }

  private[lamp] def makeModel(
      dataLayout: Seq[Metadata],
      dropout: Double,
      hiddenSize: Int,
      outputSize: Int,
      softmax: Boolean,
      selectFirstOutputDimension: Boolean,
      modelTensorOptions: TensorOptions
  )(implicit pool: AllocatedVariablePool) = {
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
      if (softmax && !selectFirstOutputDimension) Fun(_.logSoftMax(dim = 1))
      else if (softmax && selectFirstOutputDimension)
        Fun(_.logSoftMax(dim = 1).select(1, 0).view(List(-1, 1)))
      else Fun(identity)
    )

    model

  }

  private[lamp] def trainAndPredictKnn(
      k: Int,
      data: Tensor,
      predictables: Tensor,
      predictionsOnBasemodels: Seq[Tensor],
      predictablesPredictionsOnBasemodels: Seq[Tensor],
      target: Tensor,
      targetType: TargetType,
      dataLayout: Seq[Metadata],
      device: Device,
      precision: FloatingPointPrecision,
      minibatchSize: Int
  ) =
    IO {

      val trainingFeatures = makeFusedFeatureMatrix(
        data,
        dataLayout,
        predictionsOnBasemodels,
        precision,
        device
      )
      val predictableFeatures = makeFusedFeatureMatrix(
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
          lamp.knn.squaredEuclideanDistance,
          minibatchSize
        )
        val indicesJvm = TensorHelpers.toLongMat(indices).map(_.toInt)
        predictableFeatures.release
        trainingFeatures.release
        val prediction = targetType match {
          case Classification(classes, _) =>
            val targetVec =
              TensorHelpers.toLongMat(target).toVec.map(_.toInt)
            lamp.knn.classification(targetVec, indicesJvm, classes, log = true)
          case Regression | ECDFRegression =>
            import lamp.syntax
            val targetVec = target.toMat.toVec
            Mat(lamp.knn.regression(targetVec, indicesJvm))
        }

        TensorHelpers.fromMat(prediction, CPU, precision),

      }

      predicteds

    }

  private[lamp] def makeFusedFeatureMatrix(
      data: Tensor,
      dataLayout: Seq[Metadata],
      predictionsOnBasemodels: Seq[Tensor],
      precision: FloatingPointPrecision,
      device: Device
  ) = {

    val (num, cat) = separateFeatures(data, dataLayout)
    val (trainNumerical, trainCategorical) =
      if (predictionsOnBasemodels.isEmpty) (num, cat)
      else {
        val numWithPredictions =
          ATen.cat(num +: predictionsOnBasemodels.toArray, 1)
        num.release
        (numWithPredictions, cat)
      }
    val categoricalOneHot = trainCategorical.map {
      case (t, numClasses) =>
        val t2 = ATen.one_hot(t, numClasses)
        val t3 = precision match {
          case SinglePrecision => ATen._cast_Float(t2, false)
          case DoublePrecision => ATen._cast_Double(t2, false)
        }

        t.release
        t2.release
        t3
    }
    val catted = ATen.cat((categoricalOneHot :+ trainNumerical).toArray, 1)
    trainNumerical.release
    categoricalOneHot.foreach(_.release)
    val onDevice = device.to(catted)
    catted.release
    onDevice

  }
  private[lamp] def trainAndPredictExtratrees(
      k: Int,
      m: Int,
      nMin: Int,
      parallelism: Int,
      data: Tensor,
      predictables: Tensor,
      predictionsOnBasemodels: Seq[Tensor],
      predictablesPredictionsOnBasemodels: Seq[Tensor],
      target: Tensor,
      targetType: TargetType,
      dataLayout: Seq[Metadata]
  ) =
    IO {
      import lamp.syntax
      val precision = TensorHelpers.precision(data).get
      val trainingFeatures = makeFusedFeatureMatrix(
        data,
        dataLayout,
        predictionsOnBasemodels,
        precision,
        CPU
      )
      val trainingFeaturesJvm = trainingFeatures.toMat
      trainingFeatures.release

      val predictableFeatures = makeFusedFeatureMatrix(
        predictables,
        dataLayout,
        predictablesPredictionsOnBasemodels,
        precision,
        CPU
      )
      val predictableFeaturesJvm = predictableFeatures.toMat
      predictableFeatures.release

      targetType match {
        case Classification(classes, _) =>
          val targetVec =
            TensorHelpers.toLongMat(target).toVec.map(_.toInt)
          val trained = lamp.extratrees.buildForestClassification(
            data = trainingFeaturesJvm,
            target = targetVec,
            numClasses = classes,
            nMin = nMin,
            k = k,
            m = m,
            parallelism = parallelism
          )
          val prediction = lamp.extratrees
            .predictClassification(trained, predictableFeaturesJvm)
            .map(v => math.log(v + 1e-6))
          val predictionT = TensorHelpers.fromMat(prediction, CPU, precision)
          (predictionT, Left(trained))

        case Regression | ECDFRegression =>
          import lamp.syntax
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
          val predictionT = TensorHelpers.fromMat(prediction, CPU, precision)
          (predictionT, Right(trained))
      }

    }

  private[lamp] def trainAndPredict1(
      data: Tensor,
      predictables: Tensor,
      predictionsOnBasemodels: Seq[Tensor],
      predictablesPredictionsOnBasemodels: Seq[Tensor],
      target: Tensor,
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
      logFrequency: Int,
      logger: Option[Logger]
  )(implicit pool: AllocatedVariablePool) =
    IO {
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
      val (lossFunction, classWeightsT) = targetType match {
        case Regression     => (LossFunctions.L1Loss, None)
        case ECDFRegression => (LossFunctions.L1Loss, None)
        case Classification(classes, classWeights) =>
          val classWeightsT =
            TensorHelpers.fromVec(classWeights.toVec, device, precision)
          (
            LossFunctions.NLL(
              numClasses = classes,
              classWeights = classWeightsT
            ),
            Some(classWeightsT)
          )
      }

      val (trainNumerical, trainCategorical) = {
        val (num, cat) = separateFeatures(data, dataLayout)
        if (predictionsOnBasemodels.isEmpty) (num, cat.map(_._1))
        else {
          val numWithPredictions =
            ATen.cat(num +: predictionsOnBasemodels.toArray, 1)
          num.release
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
          device = device
        )

      val batchesInEpoch = miniBatches._2
      val maxEpochs = epochs.max

      val modelWithOptimizer =
        SupervisedModel(model, lossFunction).zipOptimizer(
          AdamW.factory(
            weightDecay = simple(weightDecay),
            learningRate = simple(learningRate),
            scheduler = LearningRateSchedule.stepAfter(
              steps = (maxEpochs * batchesInEpoch * 0.8).toLong,
              factor = 0.1
            )
          )
        )

      def batchStream = miniBatches._1

      val trainingCallback = TrainingCallback.noop

      val (predictNumerical, predictCategorical) = {
        val (num, cat) = separateFeatures(predictables, dataLayout)
        if (predictablesPredictionsOnBasemodels.isEmpty) (num, cat.map(_._1))
        else {
          val numWithPredictions =
            ATen.cat(num +: predictablesPredictionsOnBasemodels.toArray, 1)
          num.release
          (numWithPredictions, cat.map(_._1))
        }
      }

      def loop(
          epoch: Int,
          predictedAtEpochs: Seq[(Int, Tensor, Seq[(Tensor, PTag)])]
      ): IO[Seq[(Int, Tensor, Seq[(Tensor, PTag)])]] =
        if (epoch > maxEpochs) {
          IO {
            logger.foreach(_.info(s"Max epochs ($maxEpochs) reached."))
            predictedAtEpochs
          }
        } else {
          for {
            _ <- IOLoops.oneEpoch(
              modelWithOptimizer,
              batchStream,
              trainingCallback,
              logger,
              logFrequency
            )
            next <- loop(
              epoch + 1, {
                logger.foreach(_.info(s"Epoch $epoch done."))
                if (epochs.contains(epoch)) {
                  val predicted = modelWithOptimizer.model.module
                    .forward(
                      (
                        predictCategorical.map(v =>
                          const(device.to(v)).releasable
                        ),
                        const(device.to(predictNumerical)).releasable
                      )
                    )
                  val copy = predicted.value.to(predictNumerical.options, true)
                  val modelState = model.state.map { v =>
                    val clone = ATen.clone(v._1.value)
                    val c1 = if (!clone.options.isCPU) {
                      val cp = CPU.to(clone)
                      clone.release
                      cp
                    } else clone
                    (c1, v._2)
                  }
                  predicted.releaseAll()
                  (epoch, copy, modelState) +: predictedAtEpochs
                } else predictedAtEpochs
              }
            )
          } yield next
        }

      val predicteds = loop(0, Nil).map { r =>
        trainNumerical.release
        trainCategorical.foreach(_.release)
        predictNumerical.release
        predictCategorical.foreach(_.release)
        classWeightsT.foreach(_.release)
        modelWithOptimizer.release
        r
      }

      predicteds

    }.flatMap(identity)

  private[lamp] def aggregatePredictionsAndModelsPerEpoch[ModelState](
      byEpoch: Seq[(Int, Seq[(Int, Tensor, ModelState, Seq[Int])])],
      expectedRows: Long
  ): Seq[(Int, Tensor, Seq[ModelState])] =
    byEpoch.map {
      case (epoch, folds) =>
        val predictionsWithInstanceIDs: Vector[(Tensor, Seq[Int])] =
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
              val locations = group.map {
                case (_, innerIdx, foldIdx) => (innerIdx, foldIdx)
              }
              val tensorsOfPredictionsOfInstance = locations.map {
                case (innerIdx, foldIdx) =>
                  ATen.select(
                    predictionsWithInstanceIDs(foldIdx)._1,
                    0,
                    innerIdx
                  )

              }
              val concat =
                ATen.stack(
                  tensorsOfPredictionsOfInstance.toArray,
                  0
                )
              val mean =
                ATen.mean_1(concat, Array(0), false)
              concat.release
              tensorsOfPredictionsOfInstance.foreach(_.release)
              (instanceIdx, mean)
          }
          .sortBy(_._1)
          .map(_._2)

        val concat =
          ATen.stack(averagedPredictionsInEpoch.toArray, 0)
        averagedPredictionsInEpoch.foreach(_.release)
        assert(
          concat.sizes.head == expectedRows,
          s"${concat.sizes.head} != $expectedRows"
        )
        assert(concat.sizes.size == 2)

        (epoch, concat, folds.map(_._3))

    }

  private[lamp] def slice(
      dataFullbatch: Tensor,
      targetFullbatch: Tensor,
      predictions: Seq[Tensor],
      trainIdx: Seq[Int],
      predictIdx: Seq[Int]
  ) = {
    val trainIdxT =
      TensorHelpers.fromLongVec(
        trainIdx.toVec.map(_.toLong),
        CPU
      )
    val predictIdxT =
      TensorHelpers.fromLongVec(
        predictIdx.toVec.map(_.toLong),
        CPU
      )
    val trainFeatures =
      ATen.index_select(dataFullbatch, 0, trainIdxT)
    val trainTarget =
      ATen.index_select(targetFullbatch, 0, trainIdxT)
    val predictableFeatures =
      ATen.index_select(dataFullbatch, 0, predictIdxT)

    val trainPredictions =
      predictions.map { t => ATen.index_select(t, 0, trainIdxT) }
    val predictablePredictions =
      predictions.map { t => ATen.index_select(t, 0, predictIdxT) }

    trainIdxT.release
    predictIdxT.release
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
      dataFullbatch: Tensor,
      targetFullbatch: Tensor,
      predictions: Seq[Tensor],
      folds: Seq[(Seq[Int], Seq[Int])],
      targetType: TargetType,
      dataLayout: Seq[Metadata],
      logger: Option[Logger]
  ) = {
    val trainedFolds = folds.zipWithIndex
      .map {
        case ((trainIdx, predictIdx), foldIdx) =>
          assert((trainIdx.toSet & predictIdx.toSet).size == 0)

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
            ).map {
              case (prediction, model) =>
                (
                  0,
                  prediction,
                  model,
                  predictIdx
                )
            }
          } yield {
            trainFeatures.release
            predictableFeatures.release
            trainTarget.release
            predictablePredictions.foreach(_.release)
            trainPredictions.foreach(_.release)
            result
          }
      }
      .toList
      .sequence

    for {
      trainedFolds <- trainedFolds
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
        ).map {
          case (epoch, prediction, models) =>
            (epoch, prediction, models.map {
              case trees =>
                ExtratreesBase(trees)
            })
        }
      }
    } yield aggregatedByEpochs
  }
  def trainAndAverageFoldsKnn(
      k: Int,
      dataFullbatch: Tensor,
      targetFullbatch: Tensor,
      predictions: Seq[Tensor],
      folds: Seq[(Seq[Int], Seq[Int])],
      targetType: TargetType,
      dataLayout: Seq[Metadata],
      device: Device,
      precision: FloatingPointPrecision,
      minibatchSize: Int,
      logger: Option[Logger]
  ) = {
    val trainedFolds = folds.zipWithIndex
      .map {
        case ((trainIdx, predictIdx), foldIdx) =>
          assert((trainIdx.toSet & predictIdx.toSet).size == 0)

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
            ).map { prediction =>
              (
                0,
                prediction,
                (k, trainFeatures, trainPredictions, trainTarget),
                predictIdx
              )
            }
          } yield {
            predictableFeatures.release
            predictablePredictions.foreach(_.release)
            result
          }
      }
      .toList
      .sequence

    for {
      trainedFolds <- trainedFolds
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
        ).map {
          case (epoch, prediction, models) =>
            (epoch, prediction, models.map {
              case (k, features, predictedFeatures, target) =>
                KnnBase(k, features, predictedFeatures, target)
            })
        }
      }
    } yield aggregatedByEpochs
  }
  def trainAndAverageFolds(
      dataFullbatch: Tensor,
      targetFullbatch: Tensor,
      predictions: Seq[Tensor],
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
      logFrequency: Int,
      logger: Option[Logger]
  )(implicit pool: AllocatedVariablePool) = {
    val trainedFolds = folds.zipWithIndex
      .map {
        case ((trainIdx, predictIdx), foldIdx) =>
          assert((trainIdx.toSet & predictIdx.toSet).size == 0)

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
              logFrequency = logFrequency,
              logger = logger
            ).map(_.map(v => (v._1, v._2, v._3, predictIdx)))
          } yield {
            trainFeatures.release
            predictableFeatures.release
            trainTarget.release
            predictablePredictions.foreach(_.release)
            trainPredictions.foreach(_.release)
            result
          }
      }
      .toList
      .sequence

    for {
      trainedFolds <- trainedFolds
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

  def computeValidationErrorsAndReleasePrediction(
      pred: Tensor,
      targetType: TargetType,
      precision: FloatingPointPrecision,
      targetFullbatch: Tensor
  )(implicit pool: AllocatedVariablePool) = {
    val (lossFunction, classWeightsT) = targetType match {
      case Regression     => (LossFunctions.L1Loss, None)
      case ECDFRegression => (LossFunctions.L1Loss, None)
      case Classification(classes, classWeights) =>
        val classWeightsT =
          TensorHelpers.fromVec(
            classWeights.toVec,
            CPU,
            precision
          )
        (
          LossFunctions.NLL(
            numClasses = classes,
            classWeights = classWeightsT
          ),
          Some(classWeightsT)
        )
    }
    val (loss, _) =
      lossFunction.apply(
        const(pred).releasable,
        targetFullbatch
      )
    val lossM = loss.toMat.raw(0)
    loss.releaseAll
    classWeightsT.foreach(_.release)
    lossM
  }

  def train(
      dataFullbatch: Tensor,
      targetFullbatch: Tensor,
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
      logFrequency: Int,
      logger: Option[Logger],
      ensembleFolds: Seq[(Seq[Int], Seq[Int])],
      prescreenHyperparameters: Boolean
  )(implicit pool: AllocatedVariablePool) = {

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

    val firstpass =
      if (hyperparameters.size > 1 && prescreenHyperparameters)
        hyperparameters
          .map {
            case (wd, dro, hiddenSize) =>
              val predictionsInFolds = trainAndAverageFolds(
                dataFullbatch = dataFullbatch,
                targetFullbatch = targetFullbatch,
                predictions = Nil,
                folds = makeCVFolds(dataFullbatch.sizes.apply(0).toInt, 2, 1),
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
                logFrequency = logFrequency,
                logger = logger
              )
              predictionsInFolds.map(_.map {
                case (epoch, prediction, models) =>
                  assert(prediction.options.isCPU())
                  assert(models.forall(_.state.forall(_.options.isCPU)))
                  (epoch, wd, dro, hiddenSize, prediction, models)
              })
          }
          .toList
          .sequence
          .map(_.flatten)
      else IO { Nil }

    def baseModels(hp: Seq[(Double, Double, Int)]) = {
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
              logFrequency = logFrequency,
              logger = logger
            )
            predictionsInFolds.map(_.map {
              case (epoch, prediction, models) =>
                assert(prediction.options.isCPU())
                assert(models.forall(_.state.forall(_.options.isCPU)))
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
          )
          predictionsInFolds.map(_.map {
            case (epoch, prediction, models) =>
              assert(prediction.options.isCPU())
              assert(models.forall(_.features.options.isCPU))
              assert(models.forall(_.target.options.isCPU))
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
            )
            predictionsInFolds.map(_.map {
              case (epoch, prediction, models) =>
                assert(prediction.options.isCPU())
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
      } yield extratrees ++ knn ++ nn

    }

    def highLevelModels(predictions: Seq[Tensor]) = {
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
                logFrequency = logFrequency,
                logger = logger
              )

              withValidationErrors <- IO {
                trainedEnsembleFolds.map {
                  case (_, pred, models) =>
                    assert(pred.options.isCPU())
                    assert(models.forall(_.state.forall(_.options.isCPU)))
                    val lossM = computeValidationErrorsAndReleasePrediction(
                      pred,
                      targetType,
                      precision,
                      targetFullbatch
                    )
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
            )

            withValidationErrors <- IO {
              trainedEnsembleFolds.map {
                case (_, pred, models) =>
                  assert(pred.options.isCPU())
                  assert(models.forall(_.features.options.isCPU))
                  assert(models.forall(_.target.options.isCPU))
                  val lossM = computeValidationErrorsAndReleasePrediction(
                    pred,
                    targetType,
                    precision,
                    targetFullbatch
                  )
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
              )

              withValidationErrors <- IO {
                trainedEnsembleFolds.map {
                  case (_, pred, models) =>
                    assert(pred.options.isCPU())
                    val lossM = computeValidationErrorsAndReleasePrediction(
                      pred,
                      targetType,
                      precision,
                      targetFullbatch
                    )
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

    for {
      firstpassPred <- firstpass
      filteredHyperparameters = {
        if (firstpassPred.isEmpty) hyperparameters
        else {
          val lossAndHp = firstpassPred
            .map {
              case (_, wd, dro, hiddenSize, prediction, models) =>
                val loss = computeValidationErrorsAndReleasePrediction(
                  prediction,
                  targetType,
                  precision,
                  targetFullbatch
                )
                models.foreach(_.state.foreach(_.release))
                loss -> ((wd, dro, hiddenSize))
            }
            .groupBy(_._2)
            .toSeq
            .map { case (_, group) => group.minBy(_._1) }
            .sortBy(_._1)
          val topLoss = lossAndHp.head._1
          val maxLoss = topLoss * 1.5
          val (accept, reject) = lossAndHp.partition(_._1 <= maxLoss)
          logger.foreach(
            _.info(
              s"Reject hyperparameters ${reject.map(_._2)} based on 2-fold runs. ${accept.size} hyperparameters remain. Sorted validation losses: ${accept
                .map(_._1)}"
            )
          )
          accept.map(_._2)
        }
      }
      pred <- baseModels(filteredHyperparameters)
      _ <- IO {
        logger.foreach(_.info(s"${pred.size} base models done."))
      }
      trainedModelsWithValidationLosses <- highLevelModels(pred.map(_._2))
      _ = {
        pred.foreach(_._2.release)
      }
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
      rejected
        .flatMap(_._2.flatMap {
          case NNBase(_, state) => state
          case KnnBase(_, features, predictedFeatures, target) =>
            List(features, target) ++ predictedFeatures
          case _ => Nil
        })
        .foreach(_.release)
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
