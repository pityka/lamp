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

  def validationOneEpoch[I, M <: GenericModule[I, Variable]](
      models: Seq[SupervisedModel[I, M]],
      validationBatches: BatchStream[I],
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
        totalExamples: Long
    ): IO[(Seq[STen], Long)] = {
      makeMultipleBatches(
        devices = devices,
        makeOne = (device: Device) =>
          validationBatches
            .nextBatch(device)
      )()
        .use { batches =>
          IO {
            batches.map { case batches =>
              batches
                .zip(modelsAsEval)
                .zip(totalLossPerModel)
                .parTraverse {
                  case (
                        ((validationSample, validationTarget), modelAsEval),
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
            }
          }.flatMap {
            case EndStream         => IO.pure(EndStream)
            case EmptyBatch        => IO.pure(EmptyBatch)
            case NonEmptyBatch(io) => io.map(NonEmptyBatch(_))
          }
        }
        .flatMap {
          case EndStream  => IO.pure((totalLossPerModel, totalExamples))
          case EmptyBatch => loop(batchCount, totalLossPerModel, totalExamples)
          case NonEmptyBatch(examples) =>
            loop(
              batchCount + 1,
              totalLossPerModel,
              totalExamples + examples
            )
        }

    }

    Scope.inResource.use { implicit scope =>
      loop(
        0,
        allocatePerModelLossAcc,
        0L
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

  def oneEpoch[I, M <: GenericModule[I, Variable]](
      epochCount: Long,
      trainingCallback: TrainingCallback,
      mainModel: ModelWithOptimizer[I, M],
      trainBatches: BatchStream[I],
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
          makeOne = (device: Device) => trainBatches.nextBatch(device)
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
                    batchCounter == 0 || batchCounter % accumulateGradientOverNBatches == 1,
                  step =
                    batchCounter > 0 && batchCounter % accumulateGradientOverNBatches == 0
                )
              )
          ),
        reduce = (b, acc) => (acc + b),
        zero = 0L
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
        fetch: () => Resource[IO, A],
        transform: (Long, A) => IO[StreamControl[B]],
        reduce: (B, B) => B,
        zero: B
    ): IO[B] = {

      def startFetch(q: Queue[IO, (A, IO[Unit])]) =
        for {
          resource <- IO(fetch())
          started <- resource.allocated.flatMap(q.offer).start
        } yield started

      def loop(
          counter: Long,
          acc: B,
          queue: Queue[IO, (A, IO[Unit])]
      ): IO[B] = {
        for {
          fetched <- queue.take
          a = fetched._1
          release = fetched._2
          _ <- startFetch(queue)
          done <- transform(counter, a)
          _ <- release
          loopDone <- done match {
            case EndStream  => IO.pure(acc)
            case EmptyBatch => loop(counter, acc, queue)
            case NonEmptyBatch(b) =>
              loop(counter + 1, reduce(b, acc), queue)
          }
        } yield loopDone
      }

      for {
        q <- Queue.bounded[IO, (A, IO[Unit])](1)
        _ <- startFetch(q)
        l <- loop(0, zero, q)
      } yield l

    }

    def copyStateFromMain() = {
      val sources = mainModel.model.module.state.map(_._1.value)
      val srcDevice = sources.head.device
      models.parTraverse { m =>
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
          (gradMain +: gradPerModel).parTraverse { case (numExample, grad) =>
            IO {
              grad.foreach(_.foreach { gradTensor =>
                gradTensor.*=(numExample.toDouble)
              })
            }
          }

        _ <- gradPerModel.parTraverse { case (_, grads) =>
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

  def makeMultipleBatches[A](
      devices: List[Device],
      makeOne: (Device) => Resource[IO, StreamControl[A]]
  ): () => Resource[IO, StreamControl[List[A]]] = {
    def allocate(device: Device) =
      for {
        resource <- IO(makeOne(device))
        started <- resource.allocated
      } yield started

    def startN = devices.parTraverseN(devices.size)(allocate)

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

    () =>
      Resource
        .make(acquire = startN)(release = unifyReleases)
        .map(unifyValues)
  }

}
