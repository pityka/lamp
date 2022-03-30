package lamp.data

import org.scalatest.funsuite.AnyFunSuite
import org.saddle._
import org.saddle.order._
import lamp.autograd._
import lamp.nn._
import lamp.nn.graph._
import lamp.{CPU, CudaDevice, DoublePrecision}
import aten.ATen
import lamp.SinglePrecision
import lamp.Scope
import lamp.STen
import lamp.saddle._
import cats.effect.unsafe.implicits.global

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
      val nodes = const(lamp.saddle.fromMat(nodesM, device, precision))
      val adj = Mat(
        Vec(0d, 1d, 1d, 1d),
        Vec(1d, 0d, 0d, 0d),
        Vec(1d, 0d, 0d, 0d),
        Vec(1d, 0d, 0d, 0d)
      )
      val edges =
        lamp.saddle.fromLongMat(
          Mat(
            Vec(0L, 1L),
            Vec(0L, 2L),
            Vec(0L, 3L)
          ).T,
          device
        )

      val aggregated =
        GCN.gcnAggregation(nodes, edges.select(1, 0), edges.select(1, 1))

      val output = aggregated.value.toMat

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
      val nodes = const(lamp.saddle.fromMat(nodesM, device, precision))
      val adj = Mat(
        Vec(0d, 1d, 1d, 1d),
        Vec(1d, 0d, 0d, 0d),
        Vec(1d, 0d, 0d, 0d),
        Vec(1d, 0d, 0d, 0d)
      )
      val edges = const(
        lamp.saddle.fromLongMat(
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
              weights = param(STen.ones(List(4, 3), tOpt)),
              bias = Some(param(STen.ones(List(1, 3), tOpt)))
            ),
            Fun(implicit scope => variable => variable.relu)
          )
        )
      )

      val graph = Graph(
        nodes,
        const(STen.scalarDouble(0d, nodes.options)), // unused
        edges.value.select(1, 0),
        edges.value.select(1, 1),
        STen.scalarLong(-1L, nodes.options) // unused
      )

      val nodeStates = module.forward(graph).nodeFeatures
      val output = nodeStates.value.toMat

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
    import scala.concurrent.duration._
    val tensorLogger =
      TensorLogger.start(1.seconds)(
        s => scribe.info(s),
        (_, _) => true,
        5000,
        60000,
        1
      )

    val device = if (cuda) CudaDevice(0) else CPU
    val precision = SinglePrecision

    Scope.root { implicit scope =>
      val (featureT, labelsT, nodeIndex, unmaskedLabels) = {
        val frame = Frame(
          scala.io.Source
            .fromInputStream(getClass.getResourceAsStream("/cora.content"))
            .getLines()
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
              0 until 7 map (i =>
                labelInt.find(_ == i).head(20)
              ) reduce (_ concat _)
            )
          labelInt.zipMapIdx((i, idx) => if (keep.contains(idx)) i else -100)
        }
        val nodeIndex = frame.rowIx.map(_._1)

        val featureT = lamp.saddle.fromMat(features, device, precision)
        val labelsT =
          lamp.saddle.fromLongVec(labelIntMasked.map(_.toLong), device)
        (featureT, labelsT, nodeIndex, labelInt)
      }
      val edges = {
        val mat = Mat(
          scala.io.Source
            .fromInputStream(getClass.getResourceAsStream("/cora.cites"))
            .getLines()
            .map { line =>
              val spl = line.split("\t")
              val key1 = spl(0)
              val key2 = spl(1)
              Vec(nodeIndex.getFirst(key1), nodeIndex.getFirst(key2))
            }
            .toList: _*
        ).T

        lamp.saddle.fromLongMat(mat.map(_.toLong), device)

      }
      val graph = GraphBatchStream.Graph(
        nodeFeatures = featureT,
        edgeFeatures = STen.zeros(List(edges.shape(0))),
        edgeI = edges.select(1, 0),
        edgeJ = edges.select(1, 1)
      )
      val trainedModel = Scope { implicit scope =>
        val classWeights = STen.ones(List(7), device.options(precision))
        val model = SupervisedModel(
          sequence(
            GCN.gcn(
              in = 1433,
              out = 128,
              tOpt = device.options(precision),
              dropout = 0.95
            ),
            GenericFun[Graph, Variable](_ => _.nodeFeatures),
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

        val makeTrainingBatch = (_: IOLoops.TrainingLoopContext) =>
          GraphBatchStream.singleLargeGraph(
            graph = graph,
            targetPerNode = labelsT
          )

        val (_, trainedModel, _, _) = IOLoops
          .withSWA(
            model = model,
            optimizerFactory = RAdam
              .factory(
                learningRate = simple(0.5),
                weightDecay = simple(5e-4d)
              ),
            trainBatchesOverEpoch = makeTrainingBatch,
            warmupEpochs = 250,
            swaEpochs = 50,
            logger = Some(scribe.Logger("sdf"))
          )
          .unsafeRunSync()

        trainedModel.module
      }

      val accuracy = {
        val input = graph.toVariable

        val output =
          trainedModel.asEval.forward(input)
        val file = java.io.File.createTempFile("dfs", ".onnx")
        lamp.onnx.serializeToFile(
          file,
          output
        ) {
          case x if x == output =>
            lamp.onnx.VariableInfo(output, "output", input = false)
          case x if x == input.nodeFeatures =>
            lamp.onnx.VariableInfo(
              input.nodeFeatures,
              "node features",
              input = true
            )
          case x if x.value.options.isSparse =>
            lamp.onnx.VariableInfo(x, "graph adj", input = true)
          case x: ConstantWithoutGrad
              if x.value.shape.head == input.nodeFeatures.shape.head =>
            lamp.onnx.VariableInfo(x, "graph degree", input = true)
        }
        println(file)

        val prediction = {
          val argm = ATen.argmax(output.value.value, 1, false)
          val r = lamp.saddle.SaddleTensorHelpers.toLongMat(argm).toVec
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

    tensorLogger.cancel()
    TensorLogger.detailAllTensors(s => scribe.info(s))
    TensorLogger.detailAllTensorOptions(s => scribe.info(s))
  }

  test1("small graph mode batchstream") { cuda =>
    Scope.root { implicit scope =>
      val device = if (cuda) CudaDevice(0) else CPU
      val precision = DoublePrecision
      val graphs = Seq(
        (
          lamp.saddle.fromMat(mat.ones(5, 2), device, precision),
          lamp.saddle.fromLongMat(
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
          lamp.saddle.fromMat(mat.zeros(5, 2), device, precision),
          lamp.saddle.fromLongMat(
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
      ).map { case (nodes, edges) =>
        GraphBatchStream.Graph(
          nodeFeatures = nodes,
          edgeFeatures = STen.zeros(List(edges.shape(0))),
          edgeI = edges.select(1, 0),
          edgeJ = edges.select(1, 1)
        )
      }
      val rng = new scala.util.Random()
      val targets = lamp.saddle.fromVec(Vec(0d, 1d), device, precision)
      val (batch, _) = GraphBatchStream
        .smallGraphStream(2, graphs.toArray, targets, Some(rng))
        .nextBatch(device, 0)
        .flatMap(_._2.allocated)
        .unsafeRunSync()
      val (batchGraph, batchTarget) =
        batch.unsafeGet
      assert(batchGraph.nodeFeatures.shape == List(10, 2))
      assert(batchGraph.edgeI.toLongVec.raw(9) >= 5)
      assert(batchGraph.edgeI.toLongVec.raw(0) < 5)
      assert(
        batchGraph.vertexPoolingIndices.toLongMat.row(0) == Vec(0L, 0L, 0L, 0L,
          0L, 1L, 1L, 1L, 1L, 1L)
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
        const(lamp.saddle.fromMat(mat.ones(10, 2), device, precision))
      val edges = const(
        lamp.saddle.fromLongMat(
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
      val graphIndices =
        lamp.saddle.fromLongVec(
          Vec(0L, 0L, 0L, 0L, 0L, 1L, 1L, 1L, 1L, 1L),
          device
        )

      val graph = Graph(
        nodes,
        const(STen.scalarDouble(0d, nodes.options)), // unused
        edges.value.select(1, 0),
        edges.value.select(1, 1),
        graphIndices
      )
      val module = GCN(
        ResidualModule(
          sequence(
            Linear(
              weights = param(STen.ones(List(2, 3), tOpt)),
              bias = Some(param(STen.ones(List(1, 3), tOpt)))
            ),
            Fun(implicit scope => _.relu)
          )
        )
      )

      val graph2 = module.forward(graph)
      val nodeStates = graph2.nodeFeatures
      assert(nodeStates.value.toMat.numRows == 10)
      assert(nodeStates.value.toMat.numCols == 3)
      nodeStates.sum.backprop()
      assert(
        module.transform.transform.m1.weights.partialDerivative.isDefined
      )

      val nodesStatesM = nodeStates.value.toMat
      val graphStates = VertexPooling.apply(graph2, VertexPooling.Mean)

      val graphStatesM = graphStates.value.toMat

      assert(graphStates.value.toMat.numRows == 2)
      assert(
        graphStatesM == Mat(
          nodesStatesM.row(0, 1, 2, 3, 4).reduceCols((v, _) => v.mean),
          nodesStatesM.row(5, 6, 7, 8, 9).reduceCols((v, _) => v.mean)
        ).T
      )

    }
  }

}
