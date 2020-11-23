package lamp.data

import org.scalatest.funsuite.AnyFunSuite
import org.saddle._
import org.saddle.order._
import lamp.autograd._
import lamp.nn._
import lamp.{CPU, CudaDevice, DoublePrecision}
import aten.ATen
import lamp.SinglePrecision
import lamp.Scope
import lamp.TensorHelpers
import lamp.STen

class GCNSuite extends AnyFunSuite {
  def test1(id: String)(fun: Boolean => Unit) = {
    test(id) { fun(false) }
    test(id + "/CUDA", CudaTest) { fun(true) }
  }

  test1("gcn aggregation") { cuda =>
    Scope.root { implicit scope =>
      val device = if (cuda) CudaDevice(0) else CPU
      val precision = DoublePrecision
      val nodesM = mat.rand(4, 4)
      val nodes = const(STen.fromMat(nodesM, device, precision))
      val adj = Mat(
        Vec(0d, 1d, 1d, 1d),
        Vec(1d, 0d, 0d, 0d),
        Vec(1d, 0d, 0d, 0d),
        Vec(1d, 0d, 0d, 0d)
      )
      val edges = const(
        STen.fromLongMat(
          Mat(
            Vec(0L, 1L),
            Vec(0L, 2L),
            Vec(0L, 3L)
          ).T,
          device
        )
      )

      val aggregated = GCN.gcnAggregation(nodes, edges)

      val output = aggregated.toMat

      val expected = {
        import org.saddle.linalg._
        import org.saddle.ops.BinOps._
        val degrees = (Vec(3d, 1d, 1d, 1d) + 1).map(v => math.pow(v, -0.5))
        val ident = mat.ident(4)
        val c =
          (adj + ident).mDiagFromLeft(degrees).mDiagFromRight(degrees)
        (c mm nodesM)
      }
      assert(output.roundTo(4) == expected.roundTo(4))
    }
  }

  test1("gcn") { cuda =>
    Scope.root { implicit scope =>
      val device = if (cuda) CudaDevice(0) else CPU
      val precision = DoublePrecision
      val tOpt = device.options(precision)
      val nodesM = mat.ident(4)
      val nodes = const(STen.fromMat(nodesM, device, precision))
      val adj = Mat(
        Vec(0d, 1d, 1d, 1d),
        Vec(1d, 0d, 0d, 0d),
        Vec(1d, 0d, 0d, 0d),
        Vec(1d, 0d, 0d, 0d)
      )
      val edges = const(
        STen.fromLongMat(
          Mat(
            Vec(0L, 1L),
            Vec(0L, 2L),
            Vec(0L, 3L)
          ).T,
          device
        )
      )

      val module = GCN(
        ResidualModule(
          sequence(
            Linear(
              weights = param(STen.ones(Array(4, 3), tOpt)),
              bias = Some(param(STen.ones(Array(1, 3), tOpt)))
            ),
            Fun(implicit scope => variable => variable.relu)
          )
        )
      )

      val (nodeStates, _) = module.forward((nodes, edges))
      val output = nodeStates.toMat

      val expected = {
        import org.saddle.linalg._
        import org.saddle.ops.BinOps._
        val degrees = (Vec(3d, 1d, 1d, 1d) + 1).map(v => math.pow(v, -0.5))
        val weight = mat.ones(4, 3)
        val bias = mat.ones(1, 3)
        val ident = mat.ident(4)
        val c =
          (adj + ident).mDiagFromLeft(degrees).mDiagFromRight(degrees)
        ((c mm nodesM mm weight) + bias).map(v => math.max(0, v))
      }
      assert(output.roundTo(4) == expected.roundTo(4))
    }
  }

  test1("cora") { cuda =>
    val device = if (cuda) CudaDevice(0) else CPU
    val precision = SinglePrecision

    Scope.root { implicit scope =>
      val (featureT, labelsT, nodeIndex, unmaskedLabels) = {
        val frame = Frame(
          scala.io.Source
            .fromInputStream(getClass.getResourceAsStream("/cora.content"))
            .getLines
            .map { line =>
              val spl = line.split("\t")
              val key = spl.head
              val content = spl.drop(1).dropRight(1).map(_.toDouble).toVec
              val label = spl.last
              ((key, label), Series(content))
            }
            .toList: _*
        ).T
        val features = frame.toMat
        val labelString = frame.rowIx.map(_._2).toVec
        val str2int = labelString.toArray.distinct.sorted.zipWithIndex.toMap
        val labelInt = labelString.map(str2int)
        val labelIntMasked = {
          val keep =
            Index(
              0 until 7 map (i => labelInt.find(_ == i).head(20)) reduce (_ concat _)
            )
          labelInt.zipMapIdx((i, idx) => if (keep.contains(idx)) i else -100)
        }
        val nodeIndex = frame.rowIx.map(_._1)

        val featureT = STen.fromMat(features, device, precision)
        val labelsT =
          STen.fromLongVec(labelIntMasked.map(_.toLong), device)
        (featureT, labelsT, nodeIndex, labelInt)
      }
      val edges = {
        val mat = Mat(
          scala.io.Source
            .fromInputStream(getClass.getResourceAsStream("/cora.cites"))
            .getLines
            .map { line =>
              val spl = line.split("\t")
              val key1 = spl(0)
              val key2 = spl(1)
              Vec(nodeIndex.getFirst(key1), nodeIndex.getFirst(key2))
            }
            .toList: _*
        ).T

        STen.fromLongMat(mat.map(_.toLong), device)

      }
      val trainedModel = Scope { implicit scope =>
        val classWeights = STen.ones(Array(7), device.options(precision))
        val model = SupervisedModel(
          sequence(
            GCN.gcn(
              in = 1433,
              out = 128,
              tOpt = device.options(precision),
              dropout = 0.95
            ),
            GenericFun[(Variable, Variable), Variable](_._1),
            Linear(
              in = 128,
              out = 7,
              tOpt = device.options(precision),
              bias = false
            ),
            Fun(implicit scope => variable => variable.logSoftMax(1))
          ),
          LossFunctions.NLL(7, classWeights, ignore = -100)
        )

        val makeTrainingBatch = () =>
          GraphBatchStream.bigGraphModeFullBatch(
            nodes = featureT,
            edges = edges,
            targetPerNode = labelsT
          )

        val (_, trainedModel) = IOLoops
          .epochs(
            model = model,
            optimizerFactory = AdamW
              .factory(
                learningRate = simple(0.001),
                weightDecay = simple(5e-3d)
              ),
            trainBatchesOverEpoch = makeTrainingBatch,
            validationBatchesOverEpoch = None,
            epochs = 150,
            trainingCallback = TrainingCallback.noop,
            validationCallback = ValidationCallback.noop,
            checkpointFile = None,
            minimumCheckpointFile = None,
            logger = None
          )
          .unsafeRunSync()

        trainedModel.module
      }

      val accuracy = {
        val output =
          trainedModel.asEval.forward((const(featureT), const(edges)))
        val prediction = {
          val argm = ATen.argmax(output.value.value, 1, false)
          val r = TensorHelpers.toLongMat(argm).toVec
          argm.release
          r
        }
        val correct =
          prediction.zipMap(unmaskedLabels)((a, b) => if (a == b) 1d else 0d)
        correct.mean2
      }
      println(accuracy)
      assert(accuracy > 0.7)
    }
  }
  test1("cora ngcn") { cuda =>
    val device = if (cuda) CudaDevice(0) else CPU
    val precision = SinglePrecision

    Scope.root { implicit scope =>
      val (featureT, labelsT, nodeIndex, unmaskedLabels) = {
        val frame = Frame(
          scala.io.Source
            .fromInputStream(getClass.getResourceAsStream("/cora.content"))
            .getLines
            .map { line =>
              val spl = line.split("\t")
              val key = spl.head
              val content = spl.drop(1).dropRight(1).map(_.toDouble).toVec
              val label = spl.last
              ((key, label), Series(content))
            }
            .toList: _*
        ).T
        val features = frame.toMat
        val labelString = frame.rowIx.map(_._2).toVec
        val str2int = labelString.toArray.distinct.sorted.zipWithIndex.toMap
        val labelInt = labelString.map(str2int)
        val labelIntMasked = {
          val keep =
            Index(
              0 until 7 map (i => labelInt.find(_ == i).head(20)) reduce (_ concat _)
            )
          labelInt.zipMapIdx((i, idx) => if (keep.contains(idx)) i else -100)
        }
        val nodeIndex = frame.rowIx.map(_._1)

        val featureT = STen.fromMat(features, device, precision)
        val labelsT =
          STen.fromLongVec(labelIntMasked.map(_.toLong), device)
        (featureT, labelsT, nodeIndex, labelInt)
      }
      val edges = {
        val mat = Mat(
          scala.io.Source
            .fromInputStream(getClass.getResourceAsStream("/cora.cites"))
            .getLines
            .map { line =>
              val spl = line.split("\t")
              val key1 = spl(0)
              val key2 = spl(1)
              Vec(nodeIndex.getFirst(key1), nodeIndex.getFirst(key2))
            }
            .toList: _*
        ).T

        STen.fromLongMat(mat.map(_.toLong), device)

      }
      val classWeights = STen.ones(Array(7), device.options(precision))

      val model = SupervisedModel(
        sequence(
          NGCN.ngcn(
            in = 1433,
            middle = 128,
            out = 7,
            tOpt = device.options(precision),
            dropout = 0.95,
            K = 3,
            r = 1,
            includeZeroOrder = false
          ),
          GenericFun[(Variable, Variable), Variable](_._1),
          Fun(implicit scope => _.logSoftMax(1))
        ),
        LossFunctions.NLL(7, classWeights, ignore = -100)
      )

      val makeTrainingBatch = () =>
        GraphBatchStream.bigGraphModeFullBatch(
          nodes = featureT,
          edges = edges,
          targetPerNode = labelsT
        )

      val (_, trainedModel) = IOLoops
        .epochs(
          model = model,
          optimizerFactory = AdamW
            .factory(
              learningRate = simple(0.001),
              weightDecay = simple(5e-3d)
            ),
          trainBatchesOverEpoch = makeTrainingBatch,
          validationBatchesOverEpoch = None,
          epochs = 100,
          trainingCallback = TrainingCallback.noop,
          validationCallback = ValidationCallback.noop,
          checkpointFile = None,
          minimumCheckpointFile = None,
          logger = None
        )
        .unsafeRunSync()

      val accuracy = {
        val output =
          trainedModel.module.asEval.forward((const(featureT), const(edges)))
        val prediction = {
          val argm = ATen.argmax(output.value.value, 1, false)
          val r = TensorHelpers.toLongMat(argm).toVec
          argm.release
          r
        }
        val correct =
          prediction.zipMap(unmaskedLabels)((a, b) => if (a == b) 1d else 0d)
        correct.mean2
      }
      println(accuracy)
      assert(accuracy > 0.72)
    }
  }

  test1("small graph mode batchstream") { cuda =>
    Scope.root { implicit scope =>
      val device = if (cuda) CudaDevice(0) else CPU
      val precision = DoublePrecision
      val graphs = Seq(
        (
          STen.fromMat(mat.ones(5, 2), device, precision),
          STen.fromLongMat(
            Mat(
              Vec(0L, 1L),
              Vec(0L, 2L),
              Vec(0L, 3L),
              Vec(0L, 4L),
              Vec(1L, 2L)
            ).T,
            device
          )
        ),
        (
          STen.fromMat(mat.zeros(5, 2), device, precision),
          STen.fromLongMat(
            Mat(
              Vec(0L, 1L),
              Vec(0L, 2L),
              Vec(0L, 3L),
              Vec(0L, 4L),
              Vec(1L, 2L)
            ).T,
            device
          )
        )
      )
      val rng = org.saddle.spire.random.rng.Cmwc5.apply
      val targets = STen.fromVec(Vec(0d, 1d), device, precision)
      val (batch, _) = GraphBatchStream
        .smallGraphMode(2, graphs.toVec, targets, device, Some(rng))
        .nextBatch
        .allocated
        .unsafeRunSync()
      val ((batchNodes, batchEdges, batchGraphIndices), batchTarget) =
        batch.get
      assert(batchNodes.shape == List(10, 2))
      assert(batchEdges.toLongMat.raw(9, 0) >= 5)
      assert(batchEdges.toLongMat.raw(0, 0) < 5)
      assert(
        batchGraphIndices.toLongMat.row(0) == Vec(0L, 0L, 0L, 0L, 0L, 1L, 1L,
          1L, 1L, 1L)
      )
      assert(batchTarget.sizes.toList == List(2))
    }
  }
  test1("forward/backward") { cuda =>
    Scope.root { implicit scope =>
      val device = if (cuda) CudaDevice(0) else CPU
      val precision = DoublePrecision
      val tOpt = device.options(precision)
      val nodes =
        const(STen.fromMat(mat.ones(10, 2), device, precision))
      val edges = const(
        STen.fromLongMat(
          Mat(
            Vec(0L, 1L),
            Vec(0L, 2L),
            Vec(0L, 3L),
            Vec(0L, 4L),
            Vec(1L, 2L),
            Vec(5L, 6L),
            Vec(5L, 7L),
            Vec(5L, 8L),
            Vec(5L, 9L)
          ).T,
          device
        )
      )
      val graphIndices = const(
        STen.fromLongVec(
          Vec(0L, 0L, 0L, 0L, 0L, 1L, 1L, 1L, 1L, 1L),
          device
        )
      )
      val module = GCN(
        ResidualModule(
          sequence(
            Linear(
              weights = param(STen.ones(Array(2, 3), tOpt)),
              bias = Some(param(STen.ones(Array(1, 3), tOpt)))
            ),
            Fun(implicit scope => _.relu)
          )
        )
      )

      val (nodeStates, _) = module.forward((nodes, edges))
      assert(nodeStates.toMat.numRows == 10)
      assert(nodeStates.toMat.numCols == 3)
      nodeStates.sum.backprop()
      assert(
        module.transform.transform.m1.weights.partialDerivative.isDefined
      )

      val readoutModule = GraphReadout(
        Passthrough(
          Linear(
            weights = param(STen.eye(3, tOpt)),
            bias = None
          )
        ),
        GraphReadout.Mean
      )

      val nodesStatesM = nodeStates.toMat

      val graphStates =
        readoutModule.forward((nodeStates, edges, graphIndices))
      val graphStatesM = graphStates.toMat

      assert(graphStates.toMat.numRows == 2)
      assert(
        graphStatesM == Mat(
          nodesStatesM.row(0, 1, 2, 3, 4).reduceCols((v, _) => v.mean),
          nodesStatesM.row(5, 6, 7, 8, 9).reduceCols((v, _) => v.mean)
        ).T
      )
      graphStates.sum.backprop()
      assert(
        readoutModule.m.m.weights.partialDerivative.isDefined
      )
    }
  }

}
