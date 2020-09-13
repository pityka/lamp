package lamp.data

import org.scalatest.funsuite.AnyFunSuite
import org.saddle._
import org.saddle.order._
import lamp.autograd._
import lamp.nn._
import lamp.{CPU, CudaDevice, DoublePrecision}
import aten.ATen
import lamp.autograd.AllocatedVariablePool
import lamp.SinglePrecision

class GCNSuite extends AnyFunSuite {

  def test1(id: String)(fun: Boolean => Unit) = {
    test(id) { fun(false) }
    test(id + "/CUDA", CudaTest) { fun(true) }
  }

  test1("gcn") { cuda =>
    implicit val pool = new AllocatedVariablePool
    val device = if (cuda) CudaDevice(0) else CPU
    val precision = DoublePrecision
    val tOpt = device.options(precision)
    val nodesM = mat.ident(4)
    val nodes = const(TensorHelpers.fromMat(nodesM, device, precision))
    val adj = Mat(
      Vec(0d, 1d, 1d, 1d),
      Vec(1d, 0d, 0d, 0d),
      Vec(1d, 0d, 0d, 0d),
      Vec(1d, 0d, 0d, 0d)
    )
    val edges = const(
      TensorHelpers.fromLongMat(
        Mat(
          Vec(0L, 1L),
          Vec(0L, 2L),
          Vec(0L, 3L),
          Vec(1L, 0L),
          Vec(2L, 0L),
          Vec(3L, 0L)
        ).T,
        device
      )
    )

    val module = GCN(
      weightsFH = param(ATen.ones(Array(4, 3), tOpt)),
      bias = param(ATen.ones(Array(1, 3), tOpt)),
      dropout = 0d,
      train = true,
      relu = true
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

  test1("citeseer") { cuda =>
    val device = if (cuda) CudaDevice(0) else CPU
    val precision = SinglePrecision

    implicit val pool = new AllocatedVariablePool

    val (featureT, labelsT, nodeIndex, unmaskedLabels) = {
      val frame = Frame(
        scala.io.Source
          .fromInputStream(getClass.getResourceAsStream("/citeseer.content"))
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
            0 until 6 map (i => labelInt.find(_ == i).head(20)) reduce (_ concat _)
          )
        labelInt.zipMapIdx((i, idx) => if (keep.contains(idx)) i else -100)
      }
      val nodeIndex = frame.rowIx.map(_._1)

      val featureT = TensorHelpers.fromMat(features, device, precision)
      val labelsT =
        TensorHelpers.fromLongVec(labelIntMasked.map(_.toLong), device)
      (featureT, labelsT, nodeIndex, labelInt)
    }
    val edges = {
      val mat = Mat(
        scala.io.Source
          .fromInputStream(getClass.getResourceAsStream("/citeseer.cites"))
          .getLines
          .map { line =>
            val spl = line.split("\t")
            val key1 = spl(0)
            val key2 = spl(1)
            val k1i = nodeIndex.getFirst(key1)
            val k2i = nodeIndex.getFirst(key2)
            if (k1i >= 0 && k1i < nodeIndex.length && k2i >= 0 && k2i < nodeIndex.length)
              Some((k1i, k2i))
            else None
          }
          .collect { case Some(x) => x }
          .map { case (a, b) => if (a < b) (a, b) else (b, a) }
          .toList
          .distinct
          .map(v => Vec(v._1, v._2)): _*
      ).T

      val c1 = mat.col(0)
      val c2 = mat.col(1)
      val symmetric = Mat(c1 concat c2, c2 concat c1)

      TensorHelpers.fromLongMat(symmetric.map(_.toLong), device)

    }
    val classWeights = ATen.ones(Array(6), device.options(precision))

    val model = SupervisedModel(
      sequence(
        GCN.apply(
          in = 3703,
          hiddenSize = 32,
          tOpt = device.options(precision),
          dropout = 0.5
        ),
        GCN.apply(
          in = 32,
          hiddenSize = 6,
          tOpt = device.options(precision),
          dropout = 0.0,
          relu = false
        ),
        GenericFun[(Variable, Variable), Variable](_._1.logSoftMax(1))
      ),
      LossFunctions.NLL(6, classWeights, ignore = -100)
    )

    val makeTrainingBatch = () =>
      GraphBatchStream.bigGraphModeFullBatch(
        nodes = featureT,
        edges = edges,
        targetPerNode = labelsT
      )

    val trainedModel = IOLoops
      .epochs(
        model = model,
        optimizerFactory = AdamW
          .factory(
            learningRate = simple(0.0005),
            weightDecay = simple(5e-4)
          ),
        trainBatchesOverEpoch = makeTrainingBatch,
        validationBatchesOverEpoch = None,
        epochs = 15,
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
        val argm = ATen.argmax(output.value, 1, false)
        val r = TensorHelpers.toLongMat(argm).toVec
        argm.release
        r
      }
      val correct =
        prediction.zipMap(unmaskedLabels)((a, b) => if (a == b) 1d else 0d)
      correct.mean2
    }
    assert(accuracy > 0.57)
  }
  test1("cora") { cuda =>
    val device = if (cuda) CudaDevice(0) else CPU
    val precision = SinglePrecision

    implicit val pool = new AllocatedVariablePool

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

      val featureT = TensorHelpers.fromMat(features, device, precision)
      val labelsT =
        TensorHelpers.fromLongVec(labelIntMasked.map(_.toLong), device)
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

      val c1 = mat.col(0)
      val c2 = mat.col(1)
      val symmetric = Mat(c1 concat c2, c2 concat c1)

      TensorHelpers.fromLongMat(symmetric.map(_.toLong), device)

    }
    val classWeights = ATen.ones(Array(7), device.options(precision))

    val model = SupervisedModel(
      sequence(
        GCN.apply(
          in = 1433,
          hiddenSize = 32,
          tOpt = device.options(precision),
          dropout = 0.5
        ),
        GCN.apply(
          in = 32,
          hiddenSize = 32,
          tOpt = device.options(precision),
          dropout = 0.5
        ),
        GCN.apply(
          in = 32,
          hiddenSize = 7,
          tOpt = device.options(precision),
          dropout = 0.0,
          relu = false
        ),
        GenericFun[(Variable, Variable), Variable](_._1.logSoftMax(1))
      ),
      LossFunctions.NLL(7, classWeights, ignore = -100)
    )

    val makeTrainingBatch = () =>
      GraphBatchStream.bigGraphModeFullBatch(
        nodes = featureT,
        edges = edges,
        targetPerNode = labelsT
      )

    val trainedModel = IOLoops
      .epochs(
        model = model,
        optimizerFactory = AdamW
          .factory(
            learningRate = simple(0.001),
            weightDecay = simple(5e-4)
          ),
        trainBatchesOverEpoch = makeTrainingBatch,
        validationBatchesOverEpoch = None,
        epochs = 20,
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
        val argm = ATen.argmax(output.value, 1, false)
        val r = TensorHelpers.toLongMat(argm).toVec
        argm.release
        r
      }
      val correct =
        prediction.zipMap(unmaskedLabels)((a, b) => if (a == b) 1d else 0d)
      correct.mean2
    }
    assert(accuracy > 0.65)
  }

  test1("small graph mode batchstream") { cuda =>
    implicit val pool = new AllocatedVariablePool
    val device = if (cuda) CudaDevice(0) else CPU
    val precision = DoublePrecision
    val graphs = Seq(
      (
        TensorHelpers.fromMat(mat.ones(5, 2), device, precision),
        TensorHelpers.fromLongMat(
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
        TensorHelpers.fromMat(mat.zeros(5, 2), device, precision),
        TensorHelpers.fromLongMat(
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
    val targets = TensorHelpers.fromVec(Vec(0d, 1d), device, precision)
    val (batch, _) = GraphBatchStream
      .smallGraphMode(2, graphs.toVec, targets, device)
      .nextBatch
      .allocated
      .unsafeRunSync()
    val ((batchNodes, batchEdges, Some(batchGraphIndices)), batchTarget) =
      batch.get
    assert(batchNodes.shape == List(10, 2))
    assert(batchEdges.toLongMat.raw(9, 0) >= 5)
    assert(batchEdges.toLongMat.raw(0, 0) < 5)
    assert(
      batchGraphIndices.toLongMat.row(0) == Vec(0L, 0L, 0L, 0L, 0L, 1L, 1L, 1L,
        1L, 1L)
    )
    assert(batchTarget.sizes.toList == List(2))

  }
  test1("forward/backward") { cuda =>
    implicit val pool = new AllocatedVariablePool
    val device = if (cuda) CudaDevice(0) else CPU
    val precision = DoublePrecision
    val tOpt = device.options(precision)
    val nodes = const(TensorHelpers.fromMat(mat.ones(10, 2), device, precision))
    val edges = const(
      TensorHelpers.fromLongMat(
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
      TensorHelpers.fromLongVec(
        Vec(0L, 0L, 0L, 0L, 0L, 1L, 1L, 1L, 1L, 1L),
        device
      )
    )
    val module = GCN(
      weightsFH = param(ATen.ones(Array(2, 3), tOpt)),
      bias = param(ATen.ones(Array(1, 3), tOpt)),
      dropout = 0d,
      train = true,
      relu = true
    )

    val (nodeStates, _) = module.forward((nodes, edges))
    assert(nodeStates.toMat.numRows == 10)
    assert(nodeStates.toMat.numCols == 3)
    nodeStates.sum.backprop()
    assert(module.weightsFH.partialDerivative.isDefined)

    val readoutModule = GraphReadout(
      GCN(
        weightsFH = param(ATen.ones(Array(3, 3), tOpt)),
        bias = param(ATen.ones(Array(1, 3), tOpt)),
        dropout = 0d,
        train = true,
        relu = true
      )
    )

    val graphStates = readoutModule.forward((nodeStates, edges, graphIndices))
    assert(graphStates.toMat.numRows == 2)
    graphStates.sum.backprop()
    assert(readoutModule.m.weightsFH.partialDerivative.isDefined)

  }

}