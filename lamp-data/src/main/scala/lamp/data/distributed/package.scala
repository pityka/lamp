package lamp.data

import lamp.nn._
import lamp.data.{LoopState => _, _}
import cats.effect._
import scribe.Logger
import lamp.autograd.Variable
import lamp.STen
import lamp.Scope
import aten.NcclComm
import lamp.CudaDevice
import lamp.STenOptions
import cats.effect.std.Queue
import cats.implicits._

package object distributed {

  /** Data parallel training loop driving multiple devices from a single process
    *
    * `modelsWithDataStreams` is sequence of models, training and validation
    * streams allocated to each devices. The streams must have the same length
    * and must not contain empty batches. Models must have the same shape.
    *
    * Once the returned suspended side effect is completed the trained model is
    * in the *first* model of `modelsWithDataStreams`.
    *
    * @param modelsWithDataStreams
    * @param optimizerFactory
    * @param epochs
    * @param checkpointState
    * @param validationFrequency
    * @param returnMinValidationLossModel
    * @param learningRateSchedule
    * @param initState
    * @param accumulateGradientOverNBatches
    * @param learningRateScheduleInitState
    * @param validationLossExponentialSmoothingFactor
    * @return
    */
  def localDataParallelTrainingLoop[I, M <: GenericModule[
    I,
    Variable
  ]: Load, LRState, BatchStreamState, BatchStreamBuffers](
      modelsWithDataStreams: Seq[
        (
            SupervisedModel[I, M],
            () => BatchStream[
              (I, STen),
              BatchStreamState,
              BatchStreamBuffers
            ],
            () => BatchStream[
              (I, STen),
              BatchStreamState,
              BatchStreamBuffers
            ]
        )
      ],
      optimizerFactory: Seq[(STen, PTag)] => Optimizer,
      maxEpochs: Int,
      checkpointState: Option[
        (LoopStateWithModelAndOptimizerData, LRState) => IO[Unit]
      ] = None,
      validationFrequency: Int = 1,
      returnMinValidationLossModel: Seq[Int] = Nil,
      learningRateSchedule: LearningRateSchedule[LRState] =
        LearningRateSchedule.noop,
      initState: Option[LoopStateWithModelAndOptimizerData] = None,
      accumulateGradientOverNBatches: Int = 1,
      learningRateScheduleInitState: Option[LRState] = None,
      validationLossExponentialSmoothingFactor: Double = 1.0
  ) = {

    val nranks = modelsWithDataStreams.size
    val gpus =
      modelsWithDataStreams.map(_._1.module.state.head._1.value.device).map {
        case CudaDevice(i) => i
        case _ =>
          throw new RuntimeException(
            "localDataParallelTrainingLoop supports solely Cuda devices"
          )
      }

    def root(comm: DistributedCommunicationRoot) =
      IO.unit *> driveDistributedTraining(
        nranks = nranks,
        gpu = gpus.head,
        controlCommunication = comm,
        model = modelsWithDataStreams.head._1,
        optimizerFactory = optimizerFactory,
        trainBatches = modelsWithDataStreams.head._2,
        validationBatches = modelsWithDataStreams.head._3,
        maxEpochs = maxEpochs,
        checkpointState = checkpointState,
        validationFrequency = validationFrequency,
        returnMinValidationLossModel = returnMinValidationLossModel,
        learningRateSchedule = learningRateSchedule,
        initState = initState,
        accumulateGradientOverNBatches = accumulateGradientOverNBatches,
        learningRateScheduleInitState = learningRateScheduleInitState,
        validationLossExponentialSmoothingFactor =
          validationLossExponentialSmoothingFactor
      )

    def nonroots(comms: Seq[DistributedCommunicationNonRoot]) =
      modelsWithDataStreams.zipWithIndex.zip(gpus).drop(1).zip(comms).map {
        case ((((model, train, valid), rank), gpu), comm) =>
          IO.unit *> followDistributedTraining(
            rank = rank,
            nranks = nranks,
            gpu = gpu,
            controlCommunication = comm,
            model = model,
            trainBatches = train,
            validationBatches = valid,
            accumulateGradientOverNBatches = accumulateGradientOverNBatches
          )
      }

    for {
      comms <- LocalCommunication.make(nranks)
      _ <- nonroots(comms._2).parSequence.start
      r <- root(comms._1)
    } yield (r)
  }

  /** Drives the distributed training loop.
    *
    * Must be called on the root rank. If nranks is > 1 then
    * followDistributedTraining must be called on the rest of the ranks.
    *
    * The batch streams across all ranks must:
    *   - not contain empty batches
    *   - have the same number of batches.
    *
    * Models across all ranks must have the same shape.
    *
    * Communication is done by two independent communication channels:
    *   - tensor data is sent via NCCL, thus NCCL's requirement for network
    *     setup applies (i.e. single private network if distributed) This method
    *     will set up and tear down the NCCL communication clique.
    *   - control messages and initial rendez-vous are using an implementation
    *     of DistributedCommunicationRoot and DistributedCommunicationNonRoot.
    *     This is a very low traffic channel, 1 message before each epoch. An
    *     Akka implementation is provided which is suitable for distributed and
    *     single-process multi-gpu settings. A within process cats effect
    *     implementation is also provided for single-process multi-gpu settings.
    *
    * When the training is complete, the best model is copied into the tensors
    * of the supplied in SupervisedModel instance.
    *
    * @param nranks
    * @param gpu
    * @param controlCommunication
    * @param model
    * @param optimizerFactory
    * @param trainBatches
    * @param validationBatches
    * @param epochs
    * @param checkpointState
    * @param validationFrequency
    * @param returnMinValidationLossModel
    * @param learningRateSchedule
    * @param prefetch
    * @param initState
    * @param accumulateGradientOverNBatches
    * @param learningRateScheduleInitState
    * @param validationLossExponentialSmoothingFactor
    * @return
    */
  def driveDistributedTraining[I, M <: GenericModule[
    I,
    Variable
  ]: Load, LRState, BatchStreamState, BatchStreamBuffers](
      nranks: Int,
      gpu: Int,
      controlCommunication: DistributedCommunicationRoot,
      model: SupervisedModel[I, M],
      optimizerFactory: Seq[(STen, PTag)] => Optimizer,
      trainBatches: () => BatchStream[
        (I, STen),
        BatchStreamState,
        BatchStreamBuffers
      ],
      validationBatches: () => BatchStream[
        (I, STen),
        BatchStreamState,
        BatchStreamBuffers
      ],
      maxEpochs: Int,
      checkpointState: Option[
        (LoopStateWithModelAndOptimizerData, LRState) => IO[Unit]
      ] = None,
      validationFrequency: Int = 1,
      returnMinValidationLossModel: Seq[Int] = Nil,
      learningRateSchedule: LearningRateSchedule[LRState] =
        LearningRateSchedule.noop,
      initState: Option[LoopStateWithModelAndOptimizerData] = None,
      accumulateGradientOverNBatches: Int = 1,
      learningRateScheduleInitState: Option[LRState] = None,
      validationLossExponentialSmoothingFactor: Double = 1.0
  ): IO[LoopState] = {
    val device = CudaDevice(gpu)
    val rank = 0

    Scope.root { implicit scope =>
      val uid = lamp.NcclUniqueId()
      val clonedModelState = model.module.state.map(_._1.value.cloneTensor)

      val ncclComm =
        IO.blocking {
          scribe.info("Waiting for clique")
          STen.ncclInitComm(nranks, rank, gpu, uid)
        }

      val optimizer = IO {
        optimizerFactory(
          model.module.parameters.map(v => (v._1.value, v._2))
        )
      }

      initState.foreach { case state =>
        model.module.load(state.model)
      }

      def trainEpoch(
          optimizer: Optimizer,
          comm: NcclComm,
          lrFactor: Double
      ): IO[Double] =
        oneEpoch(
          model = model,
          stepOptimizerFn = Some({ gradients =>
            optimizer.step(gradients, lrFactor)
          }),
          batches = trainBatches(),
          logger = Some(scribe.Logger()),
          accumulateGradientOverNBatches = accumulateGradientOverNBatches,
          ncclComm = comm,
          rootRank = 0,
          device = device,
          forwardOnly = false
        )
      def validEpoch(comm: NcclComm): IO[Double] =
        oneEpoch(
          model = model,
          stepOptimizerFn = None,
          batches = validationBatches(),
          logger = Some(scribe.Logger()),
          accumulateGradientOverNBatches = accumulateGradientOverNBatches,
          ncclComm = comm,
          rootRank = 0,
          device = device,
          forwardOnly = true
        )

      def trainEpochOnCompleteClique(
          optimizer: Optimizer,
          comm: NcclComm,
          lrFactor: Double
      ) =
        controlCommunication.broadcast(
          DistributedCommunication.Train
        ) *> trainEpoch(optimizer, comm, lrFactor)

      def validationEpochOnCompleteClique(
          comm: NcclComm
      ) =
        controlCommunication.broadcast(
          DistributedCommunication.Valid
        ) *> validEpoch(comm)

      for {
        _ <- controlCommunication.onUniqueIdReady(uid)
        _ <- IO { scribe.info("Nccl unique id is ready.") }
        ncclComm <- ncclComm
        _ <- IO { scribe.info("Nccl clique is ready.") }
        optimizer <- optimizer
        _ = {
          initState.foreach { case state =>
            optimizer.load(state.optimizer)
          }
        }
        modelIsSaved <- Ref.of[IO, Boolean](false)
        r <- epochs[LRState](
          maxEpochs = maxEpochs,
          trainEpoch = (learningRateFactor: Double) =>
            trainEpochOnCompleteClique(
              optimizer,
              ncclComm,
              learningRateFactor
            ),
          validationFrequency = validationFrequency,
          returnMinValidationLossModel = returnMinValidationLossModel,
          learningRateSchedule = learningRateSchedule,
          initState = initState.map(_.loopState),
          learningRateScheduleInitState = learningRateScheduleInitState,
          validationLossExponentialSmoothingFactor =
            validationLossExponentialSmoothingFactor,
          validationEpoch = Some(
            validationEpochOnCompleteClique(
              ncclComm
            )
          ),
          checkpointState = checkpointState.map { fun =>
            val f2 = (ls: LoopState, lr: LRState) => {
              val both = LoopStateWithModelAndOptimizerData(
                ls,
                model.module.state.map(_._1.value),
                optimizer.state,
                clonedModelState
              )
              fun(both, lr)
            }
            f2
          },
          saveMinValidationLossModel = IO {
            model.module.state.map(_._1.value).zip(clonedModelState).foreach {
              case (src, dst) =>
                dst.copyFrom(src)
            }
          } *> modelIsSaved.update(_ => true)
        )
        _ <- IO { scribe.info("Broadcast stop command.") }
        _ <- controlCommunication.broadcast(
          DistributedCommunication.Stop
        )
        _ <- modelIsSaved.get.map { modelIsSaved =>
          // copying back best model
          if (modelIsSaved) {
            model.module.state.map(_._1.value).zip(clonedModelState).foreach {
              case (dst, src) =>
                dst.copyFrom(src)
            }
          }
        }
        _ <- IO(ncclComm.comm_destroy())
        _ <- IO { scribe.info("Destroyed nccl communicator object.") }
      } yield r

    }

  }

  /** Follows a distributed training loop. See the documentation of
    * driveDistributedTraining.
    *
    * @param rank
    * @param nranks
    * @param gpu
    * @param controlCommunication
    * @param model
    * @param trainBatches
    * @param validationBatches
    * @param accumulateGradientOverNBatches
    * @return
    */
  def followDistributedTraining[I, M <: GenericModule[
    I,
    Variable
  ], LRState, BatchStreamState, BatchStreamBuffers](
      rank: Int,
      nranks: Int,
      gpu: Int,
      controlCommunication: DistributedCommunicationNonRoot,
      model: SupervisedModel[I, M],
      trainBatches: () => BatchStream[
        (I, STen),
        BatchStreamState,
        BatchStreamBuffers
      ],
      validationBatches: () => BatchStream[
        (I, STen),
        BatchStreamState,
        BatchStreamBuffers
      ],
      accumulateGradientOverNBatches: Int = 1
  ): IO[Unit] = {
    val device = CudaDevice(gpu)
    assert(rank >= 1, "Follower can't have a rank of 0 ")

    def getCommandQueueAndJoinControlClique = for {
      q <- Queue.bounded[IO, DistributedCommunication.Command](1)
      uid <- controlCommunication.join(q)
    } yield (q, uid)

    def joinNcclClique(id: lamp.NcclUniqueId) = IO.blocking(
      STen.ncclInitComm(
        nRanks = nranks,
        myRank = rank,
        myDevice = gpu,
        ncclUniqueId = id
      )
    )

    def trainEpoch(comm: NcclComm): IO[Double] =
      oneEpoch(
        model = model,
        stepOptimizerFn = None,
        batches = trainBatches(),
        logger = Some(scribe.Logger()),
        accumulateGradientOverNBatches = accumulateGradientOverNBatches,
        ncclComm = comm,
        rootRank = 0,
        device = device,
        forwardOnly = false
      )
    def validEpoch(comm: NcclComm): IO[Double] =
      oneEpoch(
        model = model,
        stepOptimizerFn = None,
        batches = validationBatches(),
        logger = Some(scribe.Logger()),
        accumulateGradientOverNBatches = accumulateGradientOverNBatches,
        ncclComm = comm,
        rootRank = 0,
        device = device,
        forwardOnly = true
      )

    def loop(
        q: Queue[IO, DistributedCommunication.Command],
        comm: NcclComm
    ): IO[Unit] =
      q.take.flatMap { command =>
        command match {
          case DistributedCommunication.Train =>
            trainEpoch(comm) *> loop(q, comm)
          case DistributedCommunication.Valid =>
            validEpoch(comm) *> loop(q, comm)
          case DistributedCommunication.Stop => IO.unit
        }
      }

    for {
      qAndUid <- getCommandQueueAndJoinControlClique
      _ <- IO { scribe.info("Joined control clique.") }
      q = qAndUid._1
      id = qAndUid._2
      comm <- joinNcclClique(id)
      _ <- IO { scribe.info("Joined Nccl clique.") }
      r <- loop(q, comm)
    } yield r

  }

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
      trainEpoch: Double => IO[Double],
      validationEpoch: Option[IO[Double]],
      saveMinValidationLossModel: IO[Unit]
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

          trainingLoss <- trainEpoch(learningRateFactor)

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
                saveMinValidationLossModel.map(_ => Some(epoch))
              else if (minValidationLoss.get > maybeValidationLoss.get._1)
                saveMinValidationLossModel.map(_ => Some(epoch))
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
      batches: BatchStream[(I, STen), S, C],
      logger: Option[Logger],
      accumulateGradientOverNBatches: Int,
      ncclComm: NcclComm,
      rootRank: Int,
      device: CudaDevice,
      forwardOnly: Boolean
  ): IO[Double] = {

    def epochLoop = Scope.inResource.use { implicit scope =>
      batches.allocateBuffers(device).use { buffers =>
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
        fetch = (s) => batches.nextBatch(device, batchLoadingBuffers, s),
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
        zeroS = batches.init
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
      Scope.root { implicit scope =>
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
      Scope.root { implicit scope =>
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
          samples = batchFeature,
          target = batchTarget,
          acc = lossAccumulator,
          zeroGrad = zeroGradBeforeComputingGradients,
          switchStream = true
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
          samples = batchFeature,
          target = batchTarget,
          acc = lossAccumulator,
          switchStream = true
        )
      // totalExamples is only correct on root rank
      val totalExamples = Scope.root { implicit scope =>
        reduceNumExamples(numExamples, batchTarget.options)
      }

      totalExamples.toLong

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
