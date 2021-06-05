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

object DataParallel {

  def validationOneEpoch[I, M <: GenericModule[I, Variable], S](
      models: Seq[SupervisedModel[I, M]],
      validationBatches: BatchStream[I, S],
      validationCallback: ValidationCallback,
      logger: Option[Logger],
      epochCount: Long
  ): IO[Double] = {
    val devices = models.map(_.module.state.head._1.value.device).toList
    val modelsAsEval = models.map(_.asEval)

    def allocatePerModelLossAcc(implicit scope: Scope) =
      (models)
        .map(model => STen.scalarDouble(0d, model.module.state.head._1.options))
        .toList

    def loop(
        batchCount: Int,
        totalLossPerModel: List[STen],
        totalExamples: Long,
        s0: S
    ): IO[(Seq[STen], Long)] = {
      makeMultipleBatches(
        devices = devices,
        makeOne = (device: Device, state: S) =>
          validationBatches
            .nextBatch(device, state)
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
                            totalLoss
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
          case (_, EndStream) => IO.pure((totalLossPerModel, totalExamples))
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
      loop(
        0,
        allocatePerModelLossAcc,
        0L,
        validationBatches.init
      ).flatMap { case (totalLossPerModel, totalExamples) =>
        val validationLoss =
          totalLossPerModel.map(_.toMat.raw(0)).sum / totalExamples
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

  def oneEpoch[I, M <: GenericModule[I, Variable], S](
      epochCount: Long,
      trainingCallback: TrainingCallback,
      mainModel: ModelWithOptimizer[I, M],
      trainBatches: BatchStream[I, S],
      logger: Option[Logger],
      learningRateScheduleFactor: Double,
      models: Seq[SupervisedModel[I, M]],
      accumulateGradientOverNBatches: Int
  ) = {

    def allocatePerModelLossAcc(implicit scope: Scope) =
      (mainModel.model +: models)
        .map(model => STen.scalarDouble(0d, model.module.state.head._1.options))
        .toList

    def loop(perModelLossAcc: List[STen]) =
      driveSynchronousLoop[StreamControl[List[(I, STen)]], Long](
        fetch = makeMultipleBatches(
          devices = perModelLossAcc.map(_.device),
          makeOne =
            (device: Device, s0: S) => trainBatches.nextBatch(device, s0)
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
                  perModelLossAcc,
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
        step: Boolean,
        zeroGrad: Boolean
    ): IO[Long] = {
      assert(batch.size == perModelLossAcc.size)
      assert(batch.size == (models.size + 1))
      for {
        _ <- copyStateFromMain()
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

    def driveSynchronousLoop[A, B](
        fetch: S => IO[(S, Resource[IO, A])],
        transform: (Long, A) => IO[StreamControl[B]],
        reduce: (B, B) => B,
        zero: B,
        zeroS: S
    ): IO[B] = {

      def startFetch(q: Queue[IO, (A, IO[Unit])], s0: S) =
        for {
          resource <- fetch(s0)
          started <- resource._2.allocated.flatMap(q.offer).start
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
            case EmptyBatch => loop(counter, acc, queue, s1)
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

    def copyStateFromMain() = {
      val sources = mainModel.model.module.state.map(_._1.value)
      val srcDevice = sources.head.device
      models.toList.parTraverseN(models.size) { m =>
        IO {
          srcDevice.withOtherStreamThenSync(true) {
            val destinations = m.module.state.map(_._1.value)
            destinations.zip(sources).foreach { case (destination, source) =>
              destination.copyFrom(source)
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
      val loopDone =
        loop(lossAcc)

      loopDone.map { numInstances =>
        val totalLoss = lossAcc.map(_.toMat.raw(0)).sum
        (totalLoss, numInstances)
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
            s"Avg training loss in epoch $epochCount over $numInstances examples: $trainingLoss (${throughput
              .formatted("%.2f")} instances/sec)"
          )
        )
      }
      _ <- IO {
        trainingCallback(epochCount, trainingLoss)
      }

    } yield trainingLoss
  }

  def makeMultipleBatches[A, S](
      devices: List[Device],
      makeOne: (Device, S) => IO[(S, Resource[IO, StreamControl[A]])]
  ): S => IO[
    (S, Resource[IO, StreamControl[List[A]]])
  ] = {

    def fold(
        s0: S,
        remaining: List[Device],
        acc: List[Resource[IO, StreamControl[A]]]
    ): IO[(S, List[Resource[IO, StreamControl[A]]])] = remaining match {
      case Nil => IO.pure((s0, acc.reverse))
      case d :: ds =>
        makeOne(d, s0).flatMap { case (s1, resource) =>
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

}
