package lamp.data

import lamp.nn._
import cats.effect._
import scribe.Logger
import lamp.autograd.Variable
import lamp.STen
import lamp.Scope
// import lamp.Device
// import lamp.BufferPair
// import lamp.NcclUniqueId
import aten.NcclComm
import lamp.CudaDevice
import lamp.STenOptions

object DistributedDataParallel {

  case class LoopState(
      epoch: Int,
      lastValidationLoss: Option[Double],
      minValidationLoss: Option[Double],
      minValidationEpoch: Option[Int],
      learningCurve: List[(Int, Double, Option[(Double, Double)])]
  )

  /** Drives multiple epochs to find the minimum of smoothed validation loss
    *
    * This method does not explicitly trains a model but assumes there is a side
    * effecting effectful function which steps through an optimizer through a
    * whole epoch worth of batches.
    *
    * @param epochs
    *   Max epochs to make
    * @param checkpointState
    *   Function to checkpoint the state managed in this loop.
    * @param validationFrequency
    *   How often (by epoch count) to calculate the validation loss
    * @param returnMinValidationLossModel
    *   In which epocchs to calculat validation loss
    * @param learningRateSchedule
    * @param initState
    *   Initial state of the validation loss management state
    * @param learningRateScheduleInitState
    *   Initial state of the learning rate state
    * @param validationLossExponentialSmoothingFactor
    *   Smoothing factor in exponential smoothing of validation loss <= 1.0
    * @param trainEpoch
    *   An effectful function which steps the optimizer over a complete epoch
    *   and returns the training loss
    * @param validationEpoch
    *   An effectful function which steps through in forward mode a complete
    *   epoch and returns the validation loss
    * @param saveModel
    *   A side effect to save the current optimizer and model states
    * @return
    *   The final loop state
    */
  def epochs[LRState](
      maxEpochs: Int,
      checkpointState: Option[(LoopState, LRState) => IO[Unit]] = None,
      validationFrequency: Int = 1,
      returnMinValidationLossModel: Seq[Int] = Nil,
      learningRateSchedule: LearningRateSchedule[LRState] =
        LearningRateSchedule.noop,
      initState: Option[LoopState] = None,
      learningRateScheduleInitState: Option[LRState] = None,
      validationLossExponentialSmoothingFactor: Double = 1.0,
      trainEpoch: LRState => IO[Double],
      validationEpoch: Option[IO[Double]],
      saveModel: IO[Unit]
  ): IO[LoopState] = {

    def loop(
        epoch: Int,
        lastValidationLoss: Option[Double],
        minValidationLoss: Option[Double],
        minValidationEpoch: Option[Int],
        learningCurve: List[(Int, Double, Option[(Double, Double)])],
        lrState: LRState
    ): IO[LoopState] = {
      val (nextLearningRateScheduleState, learningRateFactor) =
        learningRateSchedule.learningRateFactor(
          state = lrState,
          epoch = epoch,
          lastValidationLoss = lastValidationLoss
        )
      if (epoch >= maxEpochs || learningRateFactor <= 0d)
        IO.pure {
          LoopState(
            epoch,
            lastValidationLoss,
            minValidationLoss,
            minValidationEpoch,
            learningCurve
          )
        }
      else {

        for {

          trainingLoss <- trainEpoch(nextLearningRateScheduleState)

          maybeValidationLoss <-
            if (epoch % validationFrequency == 0 && validationEpoch.isDefined) {
              val validationLossInThisEpoch: IO[Double] = validationEpoch.get

              validationLossInThisEpoch.map { unsmoothedValidationLoss =>
                val s = lastValidationLoss.getOrElse(unsmoothedValidationLoss)
                val smoothedValidationLoss =
                  unsmoothedValidationLoss * validationLossExponentialSmoothingFactor + s * (1d - validationLossExponentialSmoothingFactor)
                Some(
                  (smoothedValidationLoss, unsmoothedValidationLoss)
                )
              }
            } else IO.pure(None)

          nextMinValidationLoss =
            if (
              maybeValidationLoss.isEmpty || !returnMinValidationLossModel
                .contains(epoch)
            )
              minValidationLoss
            else if (minValidationLoss.isEmpty) maybeValidationLoss.map {
              case (smoothedValidationLoss, _) => smoothedValidationLoss
            }
            else
              Some(math.min(minValidationLoss.get, maybeValidationLoss.get._1))

          nextMinValidationEpoch <-
            if (
              returnMinValidationLossModel
                .contains(epoch)
            ) {
              if (maybeValidationLoss.isEmpty) IO.pure(minValidationEpoch)
              else if (minValidationLoss.isEmpty)
                saveModel.map(_ => Some(epoch))
              else if (minValidationLoss.get > maybeValidationLoss.get._1)
                saveModel.map(_ => Some(epoch))
              else IO.pure(minValidationEpoch)
            } else IO.pure(minValidationEpoch)
          nextLearningCurve = (
            epoch,
            trainingLoss,
            maybeValidationLoss
          ) :: learningCurve
          _ <-
            if (checkpointState.isDefined)
              checkpointState.get(
                LoopState(
                  epoch + 1,
                  maybeValidationLoss.map { case (smoothedValidationLoss, _) =>
                    smoothedValidationLoss
                  },
                  nextMinValidationLoss,
                  nextMinValidationEpoch,
                  nextLearningCurve
                ),
                lrState
              )
            else IO.unit
          next <- loop(
            epoch = epoch + 1,
            lastValidationLoss = maybeValidationLoss.map {
              case (smoothedValidationLoss, _) => smoothedValidationLoss
            },
            minValidationLoss = nextMinValidationLoss,
            minValidationEpoch = nextMinValidationEpoch,
            learningCurve = nextLearningCurve,
            lrState = nextLearningRateScheduleState
          )
        } yield next
      }
    }

    initState match {
      case None =>
        loop(0, None, None, None, Nil, learningRateSchedule.init)
      case Some(state) =>
        loop(
          epoch = state.epoch,
          lastValidationLoss = state.lastValidationLoss,
          minValidationLoss = state.minValidationLoss,
          minValidationEpoch = state.minValidationEpoch,
          learningCurve = state.learningCurve,
          lrState =
            learningRateScheduleInitState.getOrElse(learningRateSchedule.init)
        )
    }

  }

  /** Drives one epoch in the clique All batch streams in all members of the
    * clique *MUST* have the same number of batches otherwise this will never
    * terminate
    */
  def oneEpoch[I, M <: GenericModule[I, Variable], S, C](
      model: SupervisedModel[I, M],
      stepOptimizerFn: Option[Seq[Option[STen]] => Unit],
      trainBatches: BatchStream[(I, STen), S, C],
      logger: Option[Logger],
      accumulateGradientOverNBatches: Int,
      ncclComm: NcclComm,
      rootRank: Int,
      device: CudaDevice,
      forwardOnly: Boolean
  ): IO[Double] = {

    def epochLoop = Scope.inResource.use { implicit scope =>
      trainBatches.allocateBuffers(device).use { buffers =>
        val lossAcc =
          STen.scalarDouble(0d, model.module.state.head._1.options)
        val loopDone =
          prefetchLoop(lossAcc, buffers)

        loopDone.map { numInstances =>
          val totalLoss = lossAcc.toDoubleArray.apply(0)
          (totalLoss, numInstances)
        }
      }
    }

    def prefetchLoop(
        lossAccumulator: STen,
        batchLoadingBuffers: C
    ) = {

      DataParallel.driveSynchronousLoop[S, StreamControl[(I, STen)], Long](
        fetch = (s) => trainBatches.nextBatch(device, batchLoadingBuffers, s),
        transform = (batchCounter, batch) =>
          IO.blocking {
            batch match {
              case EmptyBatch => ???
              case EndStream  => EndStream
              case NonEmptyBatch((features, target)) =>
                NonEmptyBatch(
                  if (forwardOnly)
                    oneForwardBatch(
                      batchFeature = features,
                      batchTarget = target,
                      lossAccumulator = lossAccumulator
                    )
                  else
                    oneBatch(
                      batchFeature = features,
                      batchTarget = target,
                      lossAccumulator = lossAccumulator,
                      zeroGradBeforeComputingGradients =
                        (batchCounter % accumulateGradientOverNBatches) == 0,
                      stepOptimizerAfterComputingGradients =
                        (batchCounter % accumulateGradientOverNBatches) == (accumulateGradientOverNBatches - 1)
                    )
                )
            }

          },
        reduce = (b, acc) => (acc + b),
        zero = 0L,
        zeroS = trainBatches.init
      )
    }

    def broadcast(): Unit = {
      val tensors = model.module.state.map(_._1.value)
      tensors.foreach { tensor =>
        STen.ncclBoadcast(List((tensor, ncclComm)))
      }
    }

    def averageGradients(
        numExamples: Long,
        gradients: Seq[Option[STen]],
        isRoot: Boolean
    ): Double = {
      Scope.leak { implicit scope =>
        val numExamplesD = numExamples.toDouble
        gradients.foreach {
          _.foreach { gradient =>
            gradient *= numExamplesD
          }
        }
        val op = gradients.find(_.isDefined).flatten.get.options
        val numExamplesT = STen.scalarDouble(numExamplesD, op)
        STen.ncclReduce(List((numExamplesT, ncclComm)), numExamplesT, rootRank)
        gradients.foreach {
          _.foreach { gradient =>
            STen.ncclReduce(List((gradient, ncclComm)), gradient, rootRank)
          }
        }
        if (isRoot) {
          gradients.foreach {
            _.foreach { gradient =>
              gradient /= numExamplesT
            }
          }
        }
        lamp.CPU.to(numExamplesT).toDoubleArray(0)
      }
    }
    def reduceNumExamples(
        numExamples: Long,
        op: STenOptions
    ): Double = {
      Scope.leak { implicit scope =>
        val numExamplesD = numExamples.toDouble

        val numExamplesT = STen.scalarDouble(numExamplesD, op)
        STen.ncclReduce(List((numExamplesT, ncclComm)), numExamplesT, rootRank)
        lamp.CPU.to(numExamplesT).toDoubleArray(0)
      }
    }

    def oneBatch(
        batchFeature: I,
        batchTarget: STen,
        lossAccumulator: STen,
        zeroGradBeforeComputingGradients: Boolean,
        stepOptimizerAfterComputingGradients: Boolean
    ): Long = {
      broadcast()
      val (numExamples, gradients) =
        model.addTotalLossAndReturnGradientsAndNumExamples(
          batchFeature,
          batchTarget,
          lossAccumulator,
          zeroGradBeforeComputingGradients
        )
      if (stepOptimizerAfterComputingGradients) {
        // totalExamples is only correct on root rank
        val totalExamples =
          averageGradients(numExamples, gradients, stepOptimizerFn.isDefined)
        if (stepOptimizerFn.isDefined) {
          stepOptimizerFn.get(gradients)
        }
        totalExamples.toLong
      } else numExamples

    }
    def oneForwardBatch(
        batchFeature: I,
        batchTarget: STen,
        lossAccumulator: STen
    ): Long = {
      broadcast()
      val numExamples =
        model.addTotalLossAndReturnNumExamples(
          batchFeature,
          batchTarget,
          lossAccumulator
        )
      if (stepOptimizerFn.isDefined) {
        // totalExamples is only correct on root rank
        val totalExamples = Scope.leak { implicit scope =>
          reduceNumExamples(numExamples, batchTarget.options)
        }

        totalExamples.toLong
      } else numExamples

    }

    for {
      t1 <- IO { System.nanoTime }
      pair <- epochLoop
      t2 <- IO { System.nanoTime }
      (totalLoss, numInstances) = pair
      trainingLoss = totalLoss / numInstances
      seconds = (t2 - t1) * 1e-9
      throughput = numInstances / seconds

      _ <- IO {
        logger.foreach(
          _.info(
            s"Avg training loss over $numInstances examples: $trainingLoss (${"%.2f"
              .format(throughput)} instances/sec)"
          )
        )
      }

    } yield trainingLoss

  }

}
