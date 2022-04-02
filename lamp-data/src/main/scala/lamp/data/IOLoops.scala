package lamp.data
import lamp.nn._
import cats.effect._
import aten.Tensor
import scribe.Logger
import lamp.autograd.Variable
import lamp.STen
import lamp.Scope
import cats.effect.std.Queue
import lamp.Movable
import lamp.Device

/** Contains a training loops and helpers around it
  *
  * The two training loops implemented here are:
  *   - [[lamp.data.IOLoops.epochs]]
  *   - [[lamp.data.IOLoops.withSWA]] implements Stochastic Weight Averaging
  */
object IOLoops {

  case class TrainingLoopContext(
      epoch: Int,
      lastValidationLoss: Option[Double],
      minValidationLoss: Option[Double]
  )
  object TrainingLoopContext {
    def empty: TrainingLoopContext = IOLoops.TrainingLoopContext(0, None, None)
  }

  def forwardAndDiscardBatchStream[I, M <: GenericModule[I, Variable], S, C](
      batchStream: BatchStream[I, S, C],
      buffers: Device => Resource[IO, C],
      model: M with GenericModule[I, Variable]
  ): IO[Unit] = {

    val device = model.state.head._1.value.device

    def loop(
        batch: Resource[IO, StreamControl[(I, STen)]],
        buffer: C,
        s0: S
    ): IO[Unit] = {
      batch.use {
        case EndStream  => IO.pure(false)
        case EmptyBatch => IO.pure(true)
        case NonEmptyBatch((x, _)) =>
          IO { Scope.root { implicit scope => model.forward(x) }; true }
      } flatMap {
        case true =>
          batchStream.nextBatch(device, buffer, s0).flatMap { case (s1, next) =>
            loop(next, buffer, s1)
          }
        case false => IO.unit
      }
    }
    buffers(device).use(buffers =>
      batchStream.nextBatch(device, buffers, batchStream.init).flatMap {
        case (s1, next) =>
          loop(next, buffers, s1)
      }
    )
  }

  def runBatchStream[I, M <: GenericModule[I, Variable], S, C](
      batchStream: BatchStream[I, S, C],
      buffers: Resource[IO, C],
      model: M with GenericModule[I, Variable]
  )(implicit scope: Scope): IO[List[STen]] = {

    val device = model.state.head._1.value.device

    def loop(
        batch: Resource[IO, StreamControl[(I, STen)]],
        acc: List[STen],
        s0: S,
        buffers: C
    ): IO[List[STen]] = {
      batch.use {
        case EndStream  => IO.pure(Left(acc.reverse))
        case EmptyBatch => IO.pure(Right(acc))
        case NonEmptyBatch((x, _)) =>
          IO {
            Right(Scope { implicit scope =>
              model.forward(x).value
            } :: acc)
          }

      } flatMap {
        case Left(acc) => IO.pure(acc)
        case Right(acc) =>
          batchStream.nextBatch(device, buffers, s0).flatMap {
            case (s1, next) =>
              loop(next, acc, s1, buffers)
          }
      }

    }
    buffers.use { buffers =>
      batchStream.nextBatch(device, buffers, batchStream.init).flatMap {
        case (s1, next) =>
          loop(next, Nil, s1, buffers)
      }
    }
  }
  def parallelRunBatchStream[I, O, M <: GenericModule[I, O], S, O2: Movable, C](
      batchStream: BatchStream[I, S, C],
      bufferPerModel: Resource[IO, List[(lamp.Device, C)]],
      models: Seq[M with GenericModule[I, O]]
  )(tx: ((I, STen), O) => O2)(implicit scope: Scope): IO[Vector[O2]] = {
    import cats.effect.syntax.all._

    def loop(
        acc: Vector[O2],
        s0: S,
        devicesWithBuffers: List[(lamp.Device, C)]
    ): IO[Vector[O2]] = {
      DataParallel
        .makeMultipleBatches(
          devices = devicesWithBuffers,
          makeOne = (device: lamp.Device, state: S, c:C) =>
            batchStream
              .nextBatch(device, c, state)
        )(s0)
        .flatMap { case (s1, resource) =>
          resource
            .use { batches =>
              batches
                .map { case batches =>
                  batches
                    .zip(models)
                    .parTraverseN(batches.size) { case (batch, model) =>
                      IO {
                        Scope { implicit scope =>
                          val forwarded =
                            model.forward(batch._1)
                          val o2 = tx(batch, forwarded)
                          o2
                        }
                      }
                    }
                } match {
                case EndStream  => IO.pure((s1, EndStream))
                case EmptyBatch => IO.pure((s1, EmptyBatch))
                case NonEmptyBatch(io) =>
                  io.map(v => (s1, NonEmptyBatch(v)))
              }
            }
        }
        .flatMap {
          case (_, EndStream) => IO.pure(acc)
          case (s1, EmptyBatch) =>
            loop(acc, s1, devicesWithBuffers)
          case (s1, NonEmptyBatch(results)) =>
            loop(acc ++ results, s1, devicesWithBuffers)
        }

    }
    bufferPerModel.use(buffers =>
      loop(
        Vector.empty[O2],
        batchStream.init,
        buffers
      )
    )
  }

  // import scala.reflect.runtime.universe._

  def withSWA[I, M <: GenericModule[
    I,
    Variable
  ]: Load, LRState: TypeTag, LRStateSWA: TypeTag, BatchStreamState, BatchStreamBuffers](
      model: SupervisedModel[I, M],
      optimizerFactory: Seq[(STen, PTag)] => Optimizer,
      trainBatchesOverEpoch: TrainingLoopContext => BatchStream[
        I,
        BatchStreamState,
        BatchStreamBuffers
      ],
      warmupEpochs: Int,
      swaEpochs: Int,
      validationBatchesOverEpoch: Option[
        TrainingLoopContext => BatchStream[
          I,
          BatchStreamState,
          BatchStreamBuffers
        ]
      ] = None,
      trainingCallback: TrainingCallback = TrainingCallback.noop,
      validationCallback: ValidationCallback = ValidationCallback.noop,
      checkpointState: Option[
        (SimpleThenSWALoopState, Either[LRState, LRStateSWA]) => IO[Unit]
      ] = None,
      logger: Option[Logger] = None,
      returnMinValidationLossModel: Seq[Int] = Nil,
      learningRateSchedule: LearningRateSchedule[LRState] =
        LearningRateSchedule.decrement(20, 0.5),
      swaLearningRateSchedule: SWA.SWALearningRateSchedule[LRStateSWA] =
        SWA.SWALearningRateSchedule.cyclic(
          minFactor = 0.01,
          maxFactor = 1d,
          cycleLength = 10
        ),
      prefetch: Boolean = false,
      dataParallelModels: Seq[SupervisedModel[I, M]] = Nil,
      initState: Option[SimpleThenSWALoopState] = None,
      accumulateGradientOverNBatches: Int = 1,
      learningRateScheduleInitState: Option[LRState] = None,
      swaLearningRateScheduleInitState: Option[LRStateSWA] = None,
      swaForwardPassAfterTraining: Boolean = true,
      validationLossExponentialSmoothingFactor: Double = 1.0
  ) = {
    for {
      warmedup <-
        initState match {
          case Some(SimpleThenSWALoopState(simple, Some(_))) =>
            IO.pure(
              (
                simple.epoch,
                model,
                simple.learningCurve,
                learningRateScheduleInitState.getOrElse(
                  learningRateSchedule.init
                ),
                simple
              )
            )

          case _ =>
            epochs(
              model,
              optimizerFactory,
              trainBatchesOverEpoch,
              validationBatchesOverEpoch,
              warmupEpochs,
              trainingCallback,
              validationCallback,
              checkpointState.map(fun =>
                (s: SimpleLoopState, lr: LRState) =>
                  fun(SimpleThenSWALoopState(s, None), Left(lr))
              ),
              // checkpointLRState,
              1,
              logger,
              returnMinValidationLossModel,
              learningRateSchedule,
              prefetch,
              dataParallelModels,
              initState.map(_.simple),
              accumulateGradientOverNBatches,
              learningRateScheduleInitState,
              validationLossExponentialSmoothingFactor =
                validationLossExponentialSmoothingFactor
            )
        }
      warmupEpochReturned = warmedup._1
      warmedupModel = warmedup._2
      warmupLearningCurve = warmedup._3
      warmupLRState = warmedup._4
      warmupLoopState = warmedup._5
      swaResult <- SWA.epochs(
        warmedupModel,
        optimizerFactory,
        trainBatchesOverEpoch,
        validationBatchesOverEpoch,
        swaEpochs,
        trainingCallback,
        validationCallback,
        checkpointState.map(fun =>
          (s: SWALoopState, lrState: LRStateSWA) =>
            fun(
              SimpleThenSWALoopState(warmupLoopState, Some(s)),
              Right(lrState)
            )
        ),
        // checkpointSWALRState,
        1,
        logger,
        swaLearningRateSchedule,
        prefetch,
        dataParallelModels,
        initState.flatMap(_.swa),
        accumulateGradientOverNBatches,
        swaLearningRateScheduleInitState match {
          case Some(x) => Some(x)
          case None if swaLearningRateSchedule.init.getClass == warmupLRState.getClass =>
            Some(warmupLRState.asInstanceOf[LRStateSWA])
          case _ => None
        },
        swaForwardPassAfterTraining
      )
    } yield {
      val swaModel = swaResult._1
      val swaLearningCurve = swaResult._2
      val m = warmupLearningCurve.map(_._1).max + 1
      val concatLearningCurve = warmupLearningCurve ++ swaLearningCurve.map {
        case (epoch, l1, l2) => (epoch + m, l1, l2.map(x => (x, x)))
      }
      (warmupEpochReturned, swaModel, concatLearningCurve, warmedupModel)
    }
  }

  def epochs[I, M <: GenericModule[
    I,
    Variable
  ]: Load, LRState, BatchStreamState, BatchStreamBuffers](
      model: SupervisedModel[I, M],
      optimizerFactory: Seq[(STen, PTag)] => Optimizer,
      trainBatchesOverEpoch: TrainingLoopContext => BatchStream[
        I,
        BatchStreamState,
        BatchStreamBuffers
      ],
      validationBatchesOverEpoch: Option[
        TrainingLoopContext => BatchStream[
          I,
          BatchStreamState,
          BatchStreamBuffers
        ]
      ],
      epochs: Int,
      trainingCallback: TrainingCallback = TrainingCallback.noop,
      validationCallback: ValidationCallback = ValidationCallback.noop,
      checkpointState: Option[(SimpleLoopState, LRState) => IO[Unit]] = None,
      validationFrequency: Int = 1,
      logger: Option[Logger] = None,
      returnMinValidationLossModel: Seq[Int] = Nil,
      learningRateSchedule: LearningRateSchedule[LRState] =
        LearningRateSchedule.noop,
      prefetch: Boolean = false,
      dataParallelModels: Seq[SupervisedModel[I, M]] = Nil,
      initState: Option[SimpleLoopState] = None,
      accumulateGradientOverNBatches: Int = 1,
      learningRateScheduleInitState: Option[LRState] = None,
      printOptimizerAllocations: Boolean = false,
      validationLossExponentialSmoothingFactor: Double = 1.0
  ): IO[
    (
        Int,
        SupervisedModel[I, M],
        List[(Int, Double, Option[(Double, Double)])],
        LRState,
        SimpleLoopState
    )
  ] = {
    val modelWithOptimizer
        : ModelWithOptimizer[I, M with GenericModule[I, Variable]] =
      model.asTraining.zipOptimizer(optimizerFactory)

    if (printOptimizerAllocations) {
      val (c, bts) = {
        val state = modelWithOptimizer.optimizer.state
        val c = state.size
        val b = state.map(_.numBytes).sum
        (c, b)
      }
      println(
        s"Optimizer allocations: $c(${"%.4f".format(bts.toDouble * 1e-9)}GB)"
      )
    }

    initState.foreach { case state =>
      modelWithOptimizer.model.module.load(state.model)
      modelWithOptimizer.optimizer.load(state.optimizer)
    }

    def loop(
        epoch: Int,
        lastValidationLoss: Option[Double],
        minValidationLoss: Option[Double],
        minValidationLossModel: Option[(Int, Seq[Tensor])],
        learningCurve: List[(Int, Double, Option[(Double, Double)])],
        lrState: LRState
    ): IO[
      (
          Int,
          SupervisedModel[I, M],
          List[(Int, Double, Option[(Double, Double)])],
          LRState,
          SimpleLoopState
      )
    ] = {
      val (nextLearningRateScheduleState, learningRateFactor) =
        learningRateSchedule.learningRateFactor(
          state = lrState,
          epoch = epoch,
          lastValidationLoss = lastValidationLoss
        )
      if (epoch >= epochs || learningRateFactor <= 0d)
        IO.pure {
          modelWithOptimizer.optimizer.release()
          val doneLoopState = SimpleLoopState(
            Nil,
            Nil,
            epoch,
            lastValidationLoss,
            minValidationLoss,
            None,
            learningCurve
          )
          minValidationLossModel match {
            case None =>
              (
                epoch - 1,
                modelWithOptimizer.model,
                learningCurve.reverse,
                nextLearningRateScheduleState,
                doneLoopState
              )
            case Some((epochOfMinValidation, state)) =>
              Scope.root { implicit scope =>
                val stateOnDevice = state.map { t => STen.owned(t) }
                model.module.load(stateOnDevice)
              }
              (
                epochOfMinValidation,
                modelWithOptimizer.model,
                learningCurve.reverse,
                nextLearningRateScheduleState,
                doneLoopState
              )
          }
        }
      else {

        def copyModel = {

          logger.foreach(_.info(s"Copying model at epoch $epoch"))
          minValidationLossModel.foreach(_._2.foreach(_.release))
          val copiedState =
            model.module.state.map(_._1.value).map { t =>
              aten.ATen.clone(t.value)
            }

          (epoch, copiedState)
        }

        for {

          trainingLoss <-
            if (dataParallelModels.isEmpty)
              oneEpoch(
                epoch,
                trainingCallback,
                modelWithOptimizer,
                trainBatchesOverEpoch(
                  TrainingLoopContext(
                    epoch,
                    lastValidationLoss,
                    minValidationLoss
                  )
                ),
                logger,
                learningRateFactor,
                prefetch,
                accumulateGradientOverNBatches
              )
            else
              DataParallel.oneEpoch(
                epoch,
                trainingCallback,
                modelWithOptimizer,
                trainBatchesOverEpoch(
                  TrainingLoopContext(
                    epoch,
                    lastValidationLoss,
                    minValidationLoss
                  )
                ),
                logger,
                learningRateFactor,
                dataParallelModels,
                accumulateGradientOverNBatches
              )

          maybeValidationLoss <-
            if (
              epoch % validationFrequency == 0 && validationBatchesOverEpoch.isDefined
            ) {
              val validationLossInThisEpoch =
                if (dataParallelModels.isEmpty)
                  validationOneEpoch(
                    model = modelWithOptimizer.model,
                    validationBatches = validationBatchesOverEpoch.get(
                      TrainingLoopContext(
                        epoch,
                        lastValidationLoss,
                        minValidationLoss
                      )
                    ),
                    validationCallback = validationCallback,
                    logger = logger,
                    epochCount = epoch
                  )
                else
                  DataParallel
                    .validationOneEpoch(
                      models = modelWithOptimizer.model +: dataParallelModels,
                      validationBatches = validationBatchesOverEpoch.get(
                        TrainingLoopContext(
                          epoch,
                          lastValidationLoss,
                          minValidationLoss
                        )
                      ),
                      validationCallback = validationCallback,
                      logger = logger,
                      epochCount = epoch
                    )

              validationLossInThisEpoch.map { unsmoothedValidationLoss =>
                val s = lastValidationLoss.getOrElse(unsmoothedValidationLoss)
                val smoothedValidationLoss =
                  unsmoothedValidationLoss * validationLossExponentialSmoothingFactor + s * (1d - validationLossExponentialSmoothingFactor)
                Some(
                  (smoothedValidationLoss, unsmoothedValidationLoss)
                )
              }
            } else IO.pure(None)

          _ <- IO {
            maybeValidationLoss.foreach { case (validationLoss, _) =>
              validationCallback.apply(epoch, validationLoss)
            }
          }

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

          nextMinValidationLossModel =
            if (
              returnMinValidationLossModel
                .contains(epoch)
            ) {
              if (maybeValidationLoss.isEmpty) minValidationLossModel
              else if (minValidationLoss.isEmpty) Some(copyModel)
              else if (minValidationLoss.get > maybeValidationLoss.get._1)
                Some(copyModel)
              else minValidationLossModel
            } else minValidationLossModel
          nextLearningCurve = (
            epoch,
            trainingLoss,
            maybeValidationLoss
          ) :: learningCurve
          _ <-
            if (checkpointState.isDefined)
              checkpointState.get(
                SimpleLoopState(
                  modelWithOptimizer.model.module.state.map(_._1.value),
                  modelWithOptimizer.optimizer.state,
                  epoch + 1,
                  maybeValidationLoss.map { case (smoothedValidationLoss, _) =>
                    smoothedValidationLoss
                  },
                  nextMinValidationLoss,
                  nextMinValidationLossModel,
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
            minValidationLossModel = nextMinValidationLossModel,
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
          minValidationLossModel = state.minValidationLossModel,
          learningCurve = state.learningCurve,
          lrState =
            learningRateScheduleInitState.getOrElse(learningRateSchedule.init)
        )
    }

  }

  def oneEpoch[I, M <: GenericModule[I, Variable], S, C](
      epochCount: Long,
      trainingCallback: TrainingCallback,
      model: ModelWithOptimizer[I, M],
      trainBatches: BatchStream[I, S, C],
      logger: Option[Logger],
      learningRateScheduleFactor: Double,
      prefetch: Boolean,
      accumulateGradientOverNBatches: Int
  ): IO[Double] = {

    val device = model.model.module.state.head._1.value.device

    def processBatch(
        elem: StreamControl[(I, STen)],
        lossAcc: STen,
        batchCount: Long
    ): StreamControl[Long] = elem.map { case (sample, target) =>
      if (accumulateGradientOverNBatches <= 1) {
        val (numInstances, gradients) =
          model.model.addTotalLossAndReturnGradientsAndNumExamples(
            sample,
            target,
            lossAcc,
            zeroGrad = true
          )

        model.optimizer.step(gradients, learningRateScheduleFactor)
        numInstances
      } else {

        val (numInstances, gradients) =
          model.model.addTotalLossAndReturnGradientsAndNumExamples(
            sample,
            target,
            lossAcc,
            zeroGrad = false
          )

        if (
          (batchCount % accumulateGradientOverNBatches) == (accumulateGradientOverNBatches - 1)
        ) {
          model.optimizer.step(gradients, learningRateScheduleFactor)
          model.model.zeroGrad()
        }

        numInstances

      }

    }

    def simpleLoop(
        lossAcc: STen,
        numInstancesAcc: Long,
        batchCount: Long,
        state0: S,
        buffers: C
    ): IO[Long] = {

      trainBatches
        .nextBatch(device, buffers, state0)
        .flatMap { case (state1, resource) =>
          resource
            .use { batch =>
              IO { (state1, processBatch(batch, lossAcc, batchCount)) }
            }

        }
        .flatMap {
          case (_, EndStream) => IO.pure(numInstancesAcc)
          case (s1, EmptyBatch) =>
            simpleLoop(lossAcc, numInstancesAcc, batchCount, s1, buffers)
          case (s1, NonEmptyBatch(numInstances)) =>
            simpleLoop(
              lossAcc,
              numInstances + numInstancesAcc,
              batchCount + 1L,
              s1,
              buffers
            )
        }

    }

    def prefetch1[A, B](
        fetch: S => IO[(S, Resource[IO, StreamControl[A]])],
        transform: (Long, StreamControl[A]) => IO[StreamControl[B]],
        reduce: (B, B) => B,
        zero: B,
        zeroS: S
    ): IO[B] = {

      def loop(
          counter: Long,
          acc: B,
          queue: Queue[IO, (StreamControl[A], IO[Unit])],
          s0: S
      ): IO[B] = {
        for {
          fetched <- queue.take
          a = fetched._1
          release = fetched._2
          pair <- fetch(s0)
          s1 = pair._1
          resource = pair._2
          _ <- resource.allocated.flatMap(queue.offer).start
          done <- transform(counter, a)
          _ <- release
          loopDone <- done match {
            case EndStream  => IO.pure(acc)
            case EmptyBatch => loop(counter, acc, queue, s1)
            case NonEmptyBatch(b) =>
              loop(counter + 1, reduce(b, acc), queue, s1)
          }
        } yield loopDone
      }

      for {
        q <- Queue.bounded[IO, (StreamControl[A], IO[Unit])](1)
        pair <- fetch(zeroS)
        s1 = pair._1
        _ <- pair._2.allocated.flatMap(q.offer).start
        l <- loop(0, zero, q, s1)
      } yield l

    }

    def prefetchLoop(
        lossAcc: STen,
        buffers: C
    ) = {

      prefetch1[(I, STen), Long](
        fetch = (s) => trainBatches.nextBatch(device, buffers, s),
        transform = (batchCounter, batch) =>
          IO {
            processBatch(batch, lossAcc, batchCounter)
          },
        reduce = (b, acc) => (acc + b),
        zero = 0L,
        zeroS = trainBatches.init
      )
    }

    val epochLoop = Scope.inResource.use { implicit scope =>
      trainBatches.allocateBuffers(device).use { buffers =>
        val lossAcc =
          STen.scalarDouble(0d, model.model.module.state.head._1.options)
        val loopDone =
          if (prefetch)
            prefetchLoop(lossAcc, buffers)
          else simpleLoop(lossAcc, 0L, 0L, trainBatches.init, buffers)

        loopDone.map { numInstances =>
          val totalLoss = lossAcc.toDoubleArray.apply(0)
          (totalLoss, numInstances)
        }
      }
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
            s"Avg training loss in epoch $epochCount over $numInstances examples: $trainingLoss (${"%.2f"
              .format(throughput)} instances/sec)"
          )
        )
      }
      _ <- IO {
        trainingCallback(epochCount, trainingLoss)
      }

    } yield trainingLoss

  }
  def validationOneEpoch[I, M <: GenericModule[I, Variable], S, C](
      model: SupervisedModel[I, M],
      validationBatches: BatchStream[I, S, C],
      validationCallback: ValidationCallback,
      logger: Option[Logger],
      epochCount: Long
  ): IO[Double] = {
    val device = model.module.state.head._1.value.device
    val modelAsEval = model.asEval

    def loop(
        batchCount: Int,
        totalLoss: STen,
        totalExamples: Long,
        s0: S,
        buffers: C
    ): IO[(STen, Long)] = {
      validationBatches
        .nextBatch(device, buffers, s0)
        .flatMap { case (s1, resource) =>
          resource.use { elem =>
            IO {
              (
                s1,
                elem.map { case (validationSample, validationTarget) =>
                  val numExamples =
                    modelAsEval.addTotalLossAndReturnNumExamples(
                      validationSample,
                      validationTarget,
                      totalLoss
                    )
                  numExamples
                }
              )
            }
          }
        }
        .flatMap {
          case (_, EndStream) => IO.pure((totalLoss, totalExamples))
          case (s1, EmptyBatch) =>
            loop(batchCount, totalLoss, totalExamples, s1, buffers)
          case (s1, NonEmptyBatch(examples)) =>
            loop(
              batchCount + 1,
              totalLoss,
              totalExamples + examples,
              s1,
              buffers
            )
        }

    }

    Scope.inResource.use { implicit scope =>
      validationBatches.allocateBuffers(device).use { buffers =>
        loop(
          0,
          STen.scalarDouble(0d, model.module.state.head._1.options),
          0L,
          validationBatches.init,
          buffers
        ).flatMap { case (totalLoss, totalExamples) =>
          val validationLoss = totalLoss.toDoubleArray.apply(0) / totalExamples
          for {
            _ <- IO {
              logger.foreach(
                _.info(
                  s"Avg validation loss in epoch $epochCount over $totalExamples examples: ${validationLoss}"
                )
              )
            }
            _ <- IO {
              validationCallback(epochCount, validationLoss)
            }

          } yield validationLoss
        }
      }
    }
  }

}
