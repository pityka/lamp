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

  def oneEpoch[I, M <: GenericModule[I, Variable]: Load](
      epochCount: Long,
      trainingCallback: TrainingCallback,
      mainModel: ModelWithOptimizer[I, M],
      trainBatches: BatchStream[I],
      logger: Option[Logger],
      learningRateScheduleFactor: Double,
      models: Seq[SupervisedModel[I, M]]
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
            _,
            batches
        ) =>
          sequence(
            batches
              .map(batches =>
                synchronousStep(
                  batches,
                  perModelLossAcc
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

    def synchronousStep(
        batch: List[(I, STen)],
        perModelLossAcc: List[STen]
    ) = {
      assert(batch.size == perModelLossAcc.size)
      assert(batch.size == (models.size + 1))
      for {
        _ <- IO { copyStateFromMain() }
        gradients <- batch
          .zip(mainModel.model +: models)
          .zip(perModelLossAcc)
          .parTraverse { case ((batch, model), lossAcc) =>
            IO { computeGradient(batch, lossAcc, model) }
          }
        _ <- IO {
          averageGradientsIntoMain(
            gradMain = gradients.head,
            gradPerModel = gradients.drop(1)
          )
          stepOptimizer(gradients.head._2)
        }
      } yield gradients.map(_._1).sum

    }

    def driveSynchronousLoop[A, B](
        fetch: () => Resource[IO, A],
        transform: (Int, A) => IO[StreamControl[B]],
        reduce: (B, B) => B,
        zero: B
    ): IO[B] = {

      def startFetch(q: Queue[IO, (A, IO[Unit])]) =
        for {
          resource <- IO(fetch())
          started <- resource.allocated.flatMap(q.offer).start
        } yield started

      def loop(
          counter: Int,
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

    def copyStateFromMain(): Unit = {
      val state = mainModel.model.module.state

      models.foreach { m =>
        m.module.load(state.map(_._1.value))
      }
    }

    def computeGradient(
        elem: (I, STen),
        lossAcc: STen,
        model: SupervisedModel[I, M]
    ): (Long, Seq[Option[STen]]) =
      model.addTotalLossAndReturnGradientsAndNumExamples(
        elem._1,
        elem._2,
        lossAcc
      )

    def averageGradientsIntoMain(
        gradMain: (Long, Seq[Option[STen]]),
        gradPerModel: Seq[(Long, Seq[Option[STen]])]
    ): Unit = {
      val totalExamples = gradPerModel.map(_._1).sum
      (gradMain +: gradPerModel).foreach { case (numExample, grad) =>
        grad.foreach(_.foreach { gradTensor =>
          gradTensor.*=(numExample.toDouble)
        })
      }
      gradPerModel.foreach { case (_, grads) =>
        assert(grads.size == gradMain._2.size)
        grads.zip(gradMain._2).foreach { case (source, main) =>
          assert(source.isEmpty == main.isEmpty)
          source.zip(main).foreach { case (source, main) =>
            Scope.root { implicit scope =>
              val sourceOnMainDevice = main.device.to(source)
              main += sourceOnMainDevice
            }
          }

        }
      }
      gradMain._2.foreach(_.foreach { grad =>
        grad *= (1d / totalExamples.toDouble)
      })
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
      pair <- epochLoop
      (totalLoss, numInstances) = pair
      trainingLoss = totalLoss / numInstances

      _ <- IO {
        logger.foreach(
          _.info(
            s"Avg training loss in epoch $epochCount over $numInstances examples: $trainingLoss"
          )
        )
      }
      _ <- IO {
        trainingCallback(epochCount, trainingLoss)
      }

    } yield trainingLoss
  }

}
