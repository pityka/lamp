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
      minibatchSize: Int = 512
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
      learningRate = learningRate,
      hiddenSizes = Seq(32, 128),
      device = device,
      precision = precision,
      minibatchSize = minibatchSize,
      logFrequency = logFrequency,
      logger = logger,
      // ensembleMinibatchSize = minibatchSize,
      // ensembleHiddenSize = 8,
      // ensembleEpochs = Seq(4, 16, 64, 256),
      // ensembleWeightDecay = 0.001,
      // ensembleDropout = 0.0,
      ensembleFolds = ensembleFolds
      // ensembleLearningRate = 0.0001
    )
  }
}

case class EnsembleModel(
    selectionModels: Seq[(Int, Seq[Tensor])],
    baseModels: Seq[(Int, Seq[Seq[Tensor]])],
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
          val t = selectionModels.head._2.head.options
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
          case (hiddenSize, averagableModels) =>
            val averagablePredictions = averagableModels.map { state =>
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
                  (categorical.map(const(_)), const(numerical))
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
          case (hiddenSize, state) =>
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
              (categorical.map(const(_)), const(numericalWithPredictions))
            )
            val copy = ATen.clone(prediction.value)
            prediction.releaseAll
            copy
        }
        val stacked = ATen.stack(averagablePredictions.toArray, 0)
        val meaned = ATen.mean_1(stacked, Array(0), false)
        averagablePredictions.foreach(_.release)
        stacked.release
        categorical.foreach(_.release)
        dataOnDevice.release
        numericalWithPredictions.release
        meaned
      }
    }(t => IO(t.release))
  }
}

object AutoLoop {

  private[lamp] def makeCVFolds(length: Int, k: Int, repeat: Int) = {
    val all = IndexIntRange(length).toVec.toSeq
    0 until repeat flatMap { _ =>
      val shuffled = scala.util.Random.shuffle(all)
      val groups = shuffled.grouped(length / k + 1).take(k).toList
      assert(all.toSet == groups.flatten.toSet)
      groups.map { holdout =>
        val set = holdout.toSet
        val training = all.filterNot(i => set.contains(i))
        (training, holdout)
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
      case (Categorical(_), idx) =>
        val selected = ATen.select(data, 1, idx)
        val long = ATen._cast_Long(selected, false)
        selected.release
        long
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
        if (predictionsOnBasemodels.isEmpty) (num, cat)
        else {
          val numWithPredictions =
            ATen.cat(num +: predictionsOnBasemodels.toArray, 1)
          num.release
          (numWithPredictions, cat)
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
        if (predictablesPredictionsOnBasemodels.isEmpty) (num, cat)
        else {
          val numWithPredictions =
            ATen.cat(num +: predictablesPredictionsOnBasemodels.toArray, 1)
          num.release
          (numWithPredictions, cat)
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
                    (clone, v._2)
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

  private[lamp] def aggregatePredictionsAndModelsPerEpoch(
      byEpoch: Seq[(Int, Seq[(Int, Tensor, Seq[(Tensor, PTag)], Seq[Int])])],
      expectedRows: Long
  ): Seq[(Int, Tensor, Seq[Seq[(Tensor, PTag)]])] =
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
            s"Ensemble folds done. Aggregating predictions.."
          )
        )
        val byEpoch = trainedFolds.flatten.groupBy {
          case (epoch, _, _, _) => epoch
        }.toSeq

        aggregatePredictionsAndModelsPerEpoch(
          byEpoch,
          dataFullbatch.sizes.head
        )
      }
    } yield aggregatedByEpochs
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
      learningRate: Double,
      device: Device,
      precision: FloatingPointPrecision,
      minibatchSize: Int,
      logFrequency: Int,
      logger: Option[Logger],
      ensembleFolds: Seq[(Seq[Int], Seq[Int])]
  )(implicit pool: AllocatedVariablePool) = {

    val hyperparameters =
      hiddenSizes.flatMap(hd =>
        weighDecays.flatMap(wd => dropouts.map(dro => (wd, dro, hd)))
      )
    val baseModels =
      hyperparameters
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
                (epoch, wd, dro, hiddenSize, prediction, models)
            })
        }
        .toList
        .sequence
        .map(_.flatten)

    def highLevelModels(predictions: Seq[Tensor]) =
      hyperparameters
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
                  case (epoch, pred, models) =>
                    val (lossFunction, classWeightsT) = targetType match {
                      case Regression     => (LossFunctions.L1Loss, None)
                      case ECDFRegression => (LossFunctions.L1Loss, None)
                      case Classification(classes, classWeights) =>
                        val classWeightsT =
                          TensorHelpers.fromVec(
                            classWeights.toVec,
                            device,
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
                    (lossM, models, epoch, wd, dro, hiddenSize)

                }
              }

            } yield withValidationErrors
        }
        .toList
        .sequence
        .map(_.flatten)

    for {
      pred <- baseModels
      _ <- IO {
        logger.foreach(_.info(s"${pred.size} base models done."))
      }
      trainedModelsWithValidationLosses <- highLevelModels(pred.map(_._5))
      _ = {
        pred.foreach(_._5.release)
      }
    } yield {
      logger.foreach(_.info("Training done."))
      val (selected, rejected) = {
        val sorted = trainedModelsWithValidationLosses.sortBy(_._1)
        val top = sorted.head._1
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
      rejected.flatMap(_._2.flatten).foreach(_._1.release)
      EnsembleModel(
        selectionModels = selected.map {
          case (_, models, _, _, _, hiddenSize) =>
            (hiddenSize, models.flatten.map(_._1))
        },
        baseModels = pred.map {
          case (_, _, _, hiddenSize, _, models) =>
            (hiddenSize, models.map(_.map(_._1)))
        },
        dataLayout = dataLayout,
        targetType = targetType,
        precision = precision,
        validationLosses = validationLosses
      )
    }
  }

}
