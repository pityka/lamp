package lamp.example.arxiv

import org.saddle._
import lamp._
import lamp.autograd._
import lamp.nn._
import lamp.data._
import cats.effect.unsafe.implicits.global

case class CliConfig(
    folder: String = "",
    cuda: Boolean = false,
    checkpointSave: Option[String] = None,
    checkpointLoad: Option[String] = None
)

object Train extends App {
  scribe.info("Logger start")
  import scopt.OParser
  val builder = OParser.builder[CliConfig]
  val parser1 = {
    import builder._
    OParser.sequence(
      opt[String]("folder").action((x, c) => c.copy(folder = x)).required(),
      opt[Unit]("gpu").action((_, c) => c.copy(cuda = true)),
      opt[String]("checkpoint-save").action((x, c) =>
        c.copy(checkpointSave = Some(x))
      ),
      opt[String]("checkpoint-load").action((x, c) =>
        c.copy(checkpointLoad = Some(x))
      )
    )

  }

  OParser.parse(parser1, args, CliConfig()) match {
    case Some(config) =>
      Scope.root { implicit scope =>
        scribe.info(s"Config: $config")
        val files =
          OgbArxivDataset.readAll(new java.io.File(config.folder))
        val edges = files._1.toMat
        val nodeYears = files._2.toMat
        val nodeFeatures = files._3.toMat
        val nodeLabels = files._4.toMat
        val trainIdx = Index(nodeYears.col(0).find(_ <= 2017))
        val validIdx = Index(nodeYears.col(0).find(_ == 2018))
        val testIdx = Index(nodeYears.col(0).find(_ >= 2019))

        val device = if (config.cuda) CudaDevice(0) else CPU
        val precision = SinglePrecision

        val nodesT =
          STen.fromFloatMat(
            nodeFeatures,
            device
          )

        val edgesT = {
          val m2 = edges.map(_.toLong)

          STen.fromLongMat(
            m2,
            device
          )
        }

        def mask(idx: Index[Int]) =
          STen.fromLongVec(
            nodeLabels
              .map(_.toLong)
              .col(0)
              .zipMapIdx((value, i) => if (idx.contains(i)) value else -100L),
            device
          )

        val trainL = mask(trainIdx)
        val validL = mask(validIdx)
        val testL = mask(testIdx)

        Scope.root { implicit scope =>
          val numClasses = 40
          val classWeights =
            STen.ones(List(numClasses), device.options(precision))

          val model = SupervisedModel(
            sequence(
              // NGCN.ngcn(
              //   in = 128,
              //   middle = 128,
              //   out = numClasses,
              //   tOpt = device.options(precision),
              //   dropout = 0.5,
              //   K = 5,
              //   r = 1
              // ),
              GCN.gcn(
                in = 128,
                out = 128,
                tOpt = device.options(precision),
                dropout = 0.5
              ),
              GCN.gcn(
                in = 128,
                out = 128,
                tOpt = device.options(precision),
                dropout = 0.5
              ),
              GCN.gcn(
                in = 128,
                out = numClasses,
                tOpt = device.options(precision),
                dropout = 0.5,
                nonLinearity = false
              ),
              GenericFun[(Variable, Variable), Variable](_ => _._1),
              Fun(scope => variable => variable.logSoftMax(1)(scope))
            ),
            LossFunctions.NLL(numClasses, classWeights, ignore = -100)
          )

          println(s"Number of parameters: ${model.module.learnableParameters}")

          val makeTrainingBatch = () =>
            GraphBatchStream.bigGraphModeFullBatch(
              nodes = nodesT,
              edges = edgesT,
              targetPerNode = trainL
            )
          val makeValidationBatch = () =>
            GraphBatchStream.bigGraphModeFullBatch(
              nodes = nodesT,
              edges = edgesT,
              targetPerNode = validL
            )

          val (_, trainedModel, _, _, _) = IOLoops
            .epochs(
              model = model,
              optimizerFactory = AdamW
                .factory(
                  learningRate = simple(0.01),
                  weightDecay = simple(1e-4)
                ),
              trainBatchesOverEpoch = makeTrainingBatch,
              validationBatchesOverEpoch = Some(makeValidationBatch),
              epochs = 500,
              logger = Some(scribe.Logger("b")),
              prefetch = true
            )
            .unsafeRunSync()

          val accuracy = {
            val output =
              trainedModel.module.asEval.forward((const(nodesT), const(edgesT)))
            val prediction = {
              val argm = output.value.argmax(1, false)
              val r =
                argm.toLongMat.toVec.take(testIdx.toVec.toArray)
              r
            }

            val correct =
              prediction.zipMap(testL.toLongVec.filter(_ != -100L))((a, b) =>
                if (a == b) 1d else 0d
              )
            correct.mean2
          }
          println("Test set accuracy: " + accuracy)
        }
      }
    case _ =>
    // arguments are bad, error message will have been displayed
  }
  scribe.info("END")
}
