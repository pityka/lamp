package lamp.data

import lamp.nn._
import cats.effect._
import scribe.Logger
import lamp.autograd.Variable
import lamp.STen
import lamp.Scope
import cats.effect.std.Queue
import cats.effect.syntax.all._
import cats.syntax.all._
import lamp.Device
import lamp.BufferPair

object DataParallel {

  def validationOneEpoch[I, M <: GenericModule[I, Variable], S, C](
      models: Seq[SupervisedModel[I, M]],
      validationBatches: BatchStream[(I,STen), S, C],
      validationCallback: ValidationCallback,
      logger: Option[Logger],
      epochCount: Long
  ): IO[Double] = {
    val modelsAsEval = models.map(_.asEval)

    def allocatePerModelLossAcc(implicit scope: Scope) =
      (models)
        .map(model => STen.scalarDouble(0d, model.module.state.head._1.options))
        .toList

    def loop(
        batchCount: Int,
        totalLossPerModel: List[(STen, C)],
        totalExamples: Long,
        s0: S
    ): IO[(Seq[STen], Long)] = {
      makeMultipleBatches(
        devices = totalLossPerModel.map(t => (t._1.device, t._2)),
        makeOne = (device: Device, state: S, c: C) =>
          validationBatches
            .nextBatch(device, c, state)
      )(s0)
        .flatMap { case (s1, resource) =>
          resource
            .use { batches =>
              batches
                .map { case batches =>
                  batches
                    .zip(modelsAsEval)
                    .zip(totalLossPerModel)
                    .parTraverseN(batches.size) {

                      case (
                            (
                              (validationSample, validationTarget),
                              modelAsEval
                            ),
                            (totalLoss, _)
                          ) =>
                        IO {
                          val numExamples =
                            modelAsEval.addTotalLossAndReturnNumExamples(
                              validationSample,
                              validationTarget,
                              totalLoss
                            )
                          numExamples
                        }
                    }
                    .map(_.sum)
                } match {
                case EndStream  => IO.pure((s1, EndStream))
                case EmptyBatch => IO.pure((s1, EmptyBatch))
                case NonEmptyBatch(io) =>
                  io.map(v => (s1, NonEmptyBatch(v)))
              }
            }
        }
        .flatMap {
          case (_, EndStream) =>
            IO.pure((totalLossPerModel.map(_._1), totalExamples))
          case (s1, EmptyBatch) =>
            loop(batchCount, totalLossPerModel, totalExamples, s1)
          case (s1, NonEmptyBatch(examples)) =>
            loop(
              batchCount + 1,
              totalLossPerModel,
              totalExamples + examples,
              s1
            )
        }

    }

    Scope.inResource.use { implicit scope =>
      val devices = models.map(_.module.state.head._1.value.device)
      import cats.implicits._
      devices
        .map(device => validationBatches.allocateBuffers(device))
        .toList
        .sequence
        .use { cs =>
          loop(
            0,
            allocatePerModelLossAcc.zip(cs),
            0L,
            validationBatches.init
          ).flatMap { case (totalLossPerModel, totalExamples) =>
            val validationLoss =
              totalLossPerModel
                .map(_.toDoubleArray.apply(0))
                .sum / totalExamples
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

  /** Updates main model in place, returns average training loss
   */
  def oneEpoch[I, M <: GenericModule[I, Variable], S, C](
      epochCount: Long,
      trainingCallback: TrainingCallback,
      mainModel: ModelWithOptimizer[I, M],
      trainBatches: BatchStream[(I,STen), S, C],
      logger: Option[Logger],
      learningRateScheduleFactor: Double,
      models: Seq[SupervisedModel[I, M]],
      accumulateGradientOverNBatches: Int
  ): IO[Double] = {

    val mainDevice = mainModel.model.module.state.head._1.value.device

    def allocatePerModelLossAcc(implicit scope: Scope) =
      (mainModel.model +: models)
        .map(model => STen.scalarDouble(0d, model.module.state.head._1.options))
        .toList

    def loop(
        perModelLossAcc: List[(STen, C)],
        deviceBuffers: List[BufferPair]
    ) =
      driveSynchronousLoop[S, StreamControl[List[(I, STen)]], Long](
        fetch = makeMultipleBatches(
          devices = perModelLossAcc.map(t => (t._1.device, t._2)),
          makeOne = (device: Device, s0: S, c: C) =>
            trainBatches.nextBatch(device, c, s0)
        ),
        transform = (
            batchCounter,
            batches
        ) =>
          sequence(
            batches
              .map(batches =>
                synchronousStep(
                  batches,
                  perModelLossAcc.map(_._1),
                  deviceBuffers,
                  zeroGrad =
                    (batchCounter % accumulateGradientOverNBatches) == 0,
                  step =
                    (batchCounter % accumulateGradientOverNBatches) == (accumulateGradientOverNBatches - 1)
                )
              )
          ),
        reduce = (b, acc) => (acc + b),
        zero = 0L,
        zeroS = trainBatches.init
      )

    def sequence[A](a: StreamControl[IO[A]]): IO[StreamControl[A]] = a match {
      case EndStream            => IO.pure(EndStream)
      case EmptyBatch           => IO.pure(EmptyBatch)
      case NonEmptyBatch(batch) => batch.map(NonEmptyBatch(_))
    }

    def synchronousStep(
        batch: List[(I, STen)],
        perModelLossAcc: List[STen],
        deviceBuffers: List[BufferPair],
        step: Boolean,
        zeroGrad: Boolean
    ): IO[Long] = {
      assert(batch.size == perModelLossAcc.size)
      assert(batch.size == (models.size + 1))
      for {
        _ <- copyStateFromMain(mainDevice, deviceBuffers)
        gradients <- batch
          .zip(mainModel.model +: models)
          .zip(perModelLossAcc)
          .parTraverse { case ((batch, model), lossAcc) =>
            IO { computeGradient(batch, lossAcc, model, zeroGrad) }
          }
        _ <-
          if (step)
            averageGradientsIntoMain(
              gradMain = gradients.head,
              gradPerModel = gradients.drop(1)
            ).flatMap(_ => IO { stepOptimizer(gradients.head._2) })
          else IO.unit
      } yield gradients.map(_._1).sum

    }

  

    def copyStateFromMain(
        mainDevice: Device,
        deviceBuffers: List[BufferPair]
    ) = {
      val sources = mainModel.model.module.state.map(_._1.value)
      models.toList.zip(deviceBuffers).parTraverseN(models.size) {
        case (destinationModel, buffers) =>
          IO {
            mainDevice.withOtherStreamThenSync(true) {

              val destinations = destinationModel.module.state.map(_._1.value)
              val device = destinationModel.module.state.head._1.value.device

              Scope.root { implicit scope =>
                val copied = device.toBatched(sources, buffers)

                destinations.zip(copied).foreach { case (destination, source) =>
                  destination.copyFrom(source)
                }
              }
            }
          }
      }
    }

    def computeGradient(
        elem: (I, STen),
        lossAcc: STen,
        model: SupervisedModel[I, M],
        zeroGrad: Boolean
    ): (Long, Seq[Option[STen]]) =
      model.addTotalLossAndReturnGradientsAndNumExamples(
        elem._1,
        elem._2,
        lossAcc,
        zeroGrad
      )

    def averageGradientsIntoMain(
        gradMain: (Long, Seq[Option[STen]]),
        gradPerModel: Seq[(Long, Seq[Option[STen]])]
    ): IO[Unit] = {
      val totalExamples = gradPerModel.map(_._1).sum + gradMain._1

      for {
        _ <-
          (gradMain +: gradPerModel).toList.parTraverseN(
            gradPerModel.size + 1
          ) { case (numExample, grad) =>
            IO {
              grad.foreach(_.foreach { gradTensor =>
                gradTensor.*=(numExample.toDouble)
              })
            }
          }

        _ <- gradPerModel.toList.parTraverseN(gradPerModel.size) {
          case (_, grads) =>
            assert(grads.size == gradMain._2.size)
            IO {
              Scope.root { implicit scope =>
                val gradientSourcesOnMainDevice =
                  grads.zip(gradMain._2).map { case (source, main) =>
                    assert(source.isEmpty == main.isEmpty)
                    source.zip(main).map { case (source, main) =>
                      val sourceOnMainDevice = main.device.to(source)
                      (main, sourceOnMainDevice)
                    }
                  }
                gradientSourcesOnMainDevice.foreach(_.foreach {
                  case (main, sourceOnMainDevice) =>
                    main += sourceOnMainDevice
                })

              }

            }
        }

        _ <- IO {
          gradMain._2.foreach(_.foreach { grad =>
            grad *= (1d / totalExamples.toDouble)
          })
        }
      } yield ()
    }

    def stepOptimizer(gradients: Seq[Option[STen]]): Unit = {
      mainModel.optimizer.step(gradients, learningRateScheduleFactor)
    }

    val epochLoop = Scope.inResource.use { implicit scope =>
      val lossAcc =
        allocatePerModelLossAcc
      val devices = lossAcc.map(_.device)
      import cats.implicits._

      val buffersForBatchStream = devices
        .map(device => trainBatches.allocateBuffers(device))
        .toList
        .sequence

      val buffersForDataParallel = Scope.inResource.map { implicit scope =>
        val size = mainModel.model.module.state.map(_._1.value.numel).sum
        val op = mainModel.model.module.state.head._1.value.options
        models.toList.map { m =>
          val device = m.module.state.head._1.value.device
          val onmain = STen.zeros(List(size), mainDevice.to(op))
          val ondevice = STen.zeros(List(size), device.to(op))
          BufferPair(source = onmain, destination = ondevice)

        }
      }

      buffersForBatchStream
        .use { cs =>
          buffersForDataParallel.use { dpBuffers =>
            val loopDone =
              loop(lossAcc.zip(cs), dpBuffers)

            loopDone.map { numInstances =>
              val totalLoss = lossAcc.map(_.toDoubleArray.apply(0)).sum
              (totalLoss, numInstances)
            }
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
            s"Avg training loss in epoch $epochCount over $numInstances examples: $trainingLoss (${"%.2f".format(throughput)} instances/sec)"
          )
        )
      }
      _ <- IO {
        trainingCallback(epochCount, trainingLoss)
      }

    } yield trainingLoss
  }

  private[lamp] def makeMultipleBatches[A, S, C](
      devices: List[(Device, C)],
      makeOne: (Device, S, C) => IO[(S, Resource[IO, StreamControl[A]])]
  ): S => IO[
    (S, Resource[IO, StreamControl[List[A]]])
  ] = {

    def fold(
        s0: S,
        remaining: List[(Device, C)],
        acc: List[Resource[IO, StreamControl[A]]]
    ): IO[(S, List[Resource[IO, StreamControl[A]]])] = remaining match {
      case Nil => IO.pure((s0, acc.reverse))
      case (d, c) :: ds =>
        makeOne(d, s0, c).flatMap { case (s1, resource) =>
          fold(s1, ds, resource :: acc)
        }
    }

    def startN(s0: S) =
      fold(s0, devices, Nil)
        .map { case (s1, resources) =>
          val started = resources.parTraverseN(devices.size)(_.allocated)
          (s1, started)
        }

    def unifyReleases(
        l: List[(StreamControl[A], IO[Unit])]
    ) = l.map(_._2).sequence.map(_ => ())

    def unifyValues(
        l: List[(StreamControl[A], IO[Unit])]
    ) = {
      val elems = l.map(_._1)
      val end = elems.exists(_ == EndStream)
      val as = elems.flatMap(_ match {
        case EndStream            => Nil
        case EmptyBatch           => Nil
        case NonEmptyBatch(batch) => List(batch)
      })
      if (end) EndStream
      else if (as.isEmpty) EmptyBatch
      else NonEmptyBatch(as)

    }

    s0 =>
      startN(s0).map { case (s1, acquires) =>
        (
          s1,
          Resource
            .make(acquire = acquires)(release = unifyReleases)
            .map(unifyValues)
        )
      }
  }

   private[lamp] def driveSynchronousLoop[S, A, B](
        fetch: S => IO[(S, Resource[IO, A])],
        transform: (Long, A) => IO[StreamControl[B]],
        reduce: (B, B) => B,
        zero: B,
        zeroS: S
    ): IO[B] = {

      def startFetch(q: Queue[IO, (A, IO[Unit])], s0: S) =
        for {
          resource <- fetch(s0)
          started <- resource._2.allocated
            .attemptTap {
              case Left(exc) =>
                IO {
                  scribe.error("Error during load", exc)
                }
              case _ => IO.unit
            }
            .flatMap(q.offer)
            .start
        } yield (resource._1, started)

      def loop(
          counter: Long,
          acc: B,
          queue: Queue[IO, (A, IO[Unit])],
          s0: S
      ): IO[B] = {
        for {
          fetched <- queue.take
          a = fetched._1
          release = fetched._2
          started <- startFetch(queue, s0)
          s1 = started._1
          done <- transform(counter, a)
          _ <- release
          loopDone <- done match {
            case EndStream  => IO.pure(acc)
            case EmptyBatch =>  loop(counter, acc, queue, s1)
            case NonEmptyBatch(b) =>
              loop(counter + 1, reduce(b, acc), queue, s1)
          }
        } yield loopDone
      }

      for {
        q <- Queue.bounded[IO, (A, IO[Unit])](1)
        started <- startFetch(q, zeroS)
        l <- loop(0, zero, q, started._1)
      } yield l

    }

}
