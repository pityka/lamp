package lamp.tabular

import org.saddle._
import org.saddle.order._
import org.saddle.ops.BinOps._
import org.scalatest.funsuite.AnyFunSuite
import lamp.autograd._

import lamp.nn._
import lamp.SinglePrecision
import lamp.CPU
import lamp.CudaDevice
import java.io.File
import lamp.data.BatchStream
import lamp.data.IOLoops
import java.io.FileOutputStream
import org.saddle.index.InnerJoin
import lamp.Scope
import lamp.STen
import lamp.STenOptions

class EndToEndClassificationSuite extends AnyFunSuite {

  def parseDataset(file: File) = {
    // org.saddle.csv.CsvPa
    val src = scala.io.Source.fromInputStream(
      new java.util.zip.GZIPInputStream(new java.io.FileInputStream(file))
    )
    val frame = org.saddle.csv.CsvParser
      .parseSourceWithHeader[Double](
        src,
        recordSeparator = "\n",
        fieldSeparator = '\t'
      )
      .right
      .get
    val features = frame.filterIx(_ != "target")
    val target = frame.firstCol("target")
    src.close
    (target, features)
  }

  val datasetroot = new File(
    "../datasets/penn-ml-benchmarks/classification/"
  )
  val datasets = datasetroot.listFiles().filter(_.isDirectory()).flatMap {
    folder =>
      val dataFile =
        folder.listFiles
          .find(f => f.isFile() && f.getName().endsWith(".tsv.gz"))
      dataFile.toList.map(f => (folder.getName, f))
  }

  def trainAndPredictPytorch(
      target: Series[Int, Double],
      features: Frame[Int, String, Double]
  ) = {
    import scala.sys.process._
    val script = scala.io.Source.fromResource("classification.py").mkString
    val tmp = File.createTempFile("classifcation", ".py")
    val os = new FileOutputStream(tmp)
    os.write(script.getBytes("UTF-8"))
    os.close
    val tmp2 = File.createTempFile("data", ".csv")
    org.saddle.csv.CsvWriter
      .writeFrameToFile(
        features.addCol(target, "target", InnerJoin),
        tmp2.getAbsolutePath()
      )
    val t1 = System.nanoTime()
    val stdout =
      s"python3 ${tmp.getAbsolutePath} --file ${tmp2.getAbsolutePath}".!!
    val t2 = System.nanoTime
    (stdout.toDouble, (t2 - t1) / 1e9)
  }

  def trainAndPredictLamp(
      target: Series[Int, Double],
      features: Frame[Int, String, Double],
      cuda: Boolean
  ) = {
    Scope.leak { implicit scope =>
      val device = if (cuda) CudaDevice(0) else CPU
      val numExamples = target.length

      val testFeatures = features.row(0 -> numExamples / 3)
      val trainFeatures = features.row(numExamples / 3 + 1 -> *)
      val testTarget = Frame(target).row(0 -> numExamples / 3).colAt(0)
      val trainTarget = Frame(target).row(numExamples / 3 + 1 -> *).colAt(0)

      val testFeaturesTensor =
        STen.fromMat(testFeatures.toMat, device, SinglePrecision)

      def mlp(dim: Int, k: Int, tOpt: STenOptions) =
        sequence(
          MLP(dim, k, List(4, 4), tOpt, dropout = 0.0),
          Fun(implicit scope => _.logSoftMax(dim = 1))
        )

      val trainFeaturesTensor =
        STen.fromMat(trainFeatures.toMat, device, SinglePrecision)
      val trainTargetTensor =
        STen
          .fromLongMat(
            Mat(trainTarget.toVec.map(_.toLong)),
            device
          )
          .squeeze

      val numClasses = target.toVec.toSeq.distinct.max.toInt + 1
      val numFeatures = features.numCols
      val classWeights =
        STen.ones(Array(numClasses), device.options(SinglePrecision))

      val model = SupervisedModel(
        mlp(numFeatures, numClasses, device.options(SinglePrecision)),
        LossFunctions.NLL(numClasses, classWeights)
      )
      val rng = org.saddle.spire.random.rng.Cmwc5.apply
      val makeTrainingBatch = () =>
        BatchStream.minibatchesFromFull(
          1024,
          false,
          trainFeaturesTensor,
          trainTargetTensor,
          device,
          rng
        )

      val trainedModel = IOLoops.epochs(
        model = model,
        optimizerFactory = AdamW
          .factory(
            learningRate = simple(0.001),
            weightDecay = simple(0.0001d)
          ),
        trainBatchesOverEpoch = makeTrainingBatch,
        validationBatchesOverEpoch = None,
        epochs = 50,
        logger = None //Some(logger)
      )
      val output = trainedModel
        .map(
          _._2.module.forward(const(testFeaturesTensor))
        )
        .unsafeRunSync
      val prediction = output.toMat.rows.map(_.argmax).toVec
      val accuracy = prediction
        .zipMap(testTarget.toVec)((a, b) => if (a.toInt == b.toInt) 1d else 0d)
        .mean2

      accuracy
    }
  }
  def trainAndPredictExtraTrees(
      target: Series[Int, Double],
      features: Frame[Int, String, Double]
  ) = {
    val numExamples = target.length

    val testFeatures = features.row(0 -> numExamples / 3)
    val trainFeatures = features.row(numExamples / 3 + 1 -> *)
    val testTarget = Frame(target).row(0 -> numExamples / 3).colAt(0)
    val trainTarget = Frame(target).row(numExamples / 3 + 1 -> *).colAt(0)

    val numClasses = target.toVec.toSeq.distinct.max.toInt + 1
    val numFeatures = features.numCols

    val trainedModel = lamp.extratrees.buildForestClassification(
      data = trainFeatures.toMat,
      target = trainTarget.toVec.map(_.toInt),
      sampleWeights = None,
      numClasses = numClasses,
      nMin = 2,
      k = math.sqrt(numFeatures).toInt + 1,
      m = 30,
      parallelism = 4
    )

    val output =
      lamp.extratrees.predictClassification(trainedModel, testFeatures.toMat)
    val prediction = output.rows.map(_.argmax).toVec
    val accuracy = prediction
      .zipMap(testTarget.toVec)((a, b) => if (a.toInt == b.toInt) 1d else 0d)
      .mean2

    accuracy
  }

  test("e2e - extratrees", SlowTest) {
    val accuracies = datasets.toList
      .map {
        case (dsName, dsFile) =>
          val (target, features) = parseDataset(dsFile)
          val inbalance = target.toVec.toSeq
            .groupBy(identity)
            .toSeq
            .map(v => (v._1, v._2.size.toDouble / target.count))
            .sortBy(_._2)
            .reverse
            .head
            ._2
          (dsName, inbalance, target.count, features.numCols, target, dsFile)
      }
      .filter {
        case (_, inbalance, length, numFeatures, target, _) =>
          inbalance < 0.6 && length > 300 && length < 20000 && numFeatures > 5 && numFeatures < 1000 && target.toVec.toSeq
            .forall(_ >= 0d)
      }
      // .take(10)
      .map {
        case (dsName, _, _, _, _, dsFile) =>
          val (target, features) = parseDataset(dsFile)
          val t1 = System.nanoTime()
          val extraTreesAccuracy =
            trainAndPredictExtraTrees(target, features)
          val t2 = System.nanoTime

          val inbalance = target.toVec.toSeq
            .groupBy(identity)
            .toSeq
            .map(v => (v._1, v._2.size.toDouble / target.count))
            .sortBy(_._2)
            .reverse
            .head
            ._2

          val r = (
            dsName,
            Series(
              "majority-class-frequency" -> inbalance,
              "extratrees-accuracy" -> extraTreesAccuracy,
              "extratrees-time" -> (t2 - t1) / 1e9
            )
          )
          // ???
          r
      }
      .toFrame
      .T
      .sortedRIx

    println(new String(org.saddle.csv.CsvWriter.writeFrameToArray(accuracies)))
    println(accuracies.stringify(100, 100))

  }
  test("e2e", SlowTest) {
    val accuracies = datasets.toList
      .map {
        case (dsName, dsFile) =>
          val (target, features) = parseDataset(dsFile)
          val inbalance = target.toVec.toSeq
            .groupBy(identity)
            .toSeq
            .map(v => (v._1, v._2.size.toDouble / target.count))
            .sortBy(_._2)
            .reverse
            .head
            ._2
          (dsName, inbalance, target.count, features.numCols, target, dsFile)
      }
      .filter {
        case (_, inbalance, length, numFeatures, target, _) =>
          inbalance < 0.6 && length > 300 && length < 20000 && numFeatures > 5 && numFeatures < 1000 && target.toVec.toSeq
            .forall(_ >= 0d)
      }
      // .take(10)
      .map {
        case (dsName, _, _, _, _, dsFile) =>
          val (target, features) = parseDataset(dsFile)
          val t1 = System.nanoTime()
          val lampAccuracy1 =
            trainAndPredictLamp(target, features, cuda = false)
          val t2 = System.nanoTime
          val lampAccuracy2 =
            trainAndPredictLamp(target, features, cuda = false)
          val t3 = System.nanoTime

          val (torchAccuracy1, torchTime1) =
            trainAndPredictPytorch(target, features)

          val inbalance = target.toVec.toSeq
            .groupBy(identity)
            .toSeq
            .map(v => (v._1, v._2.size.toDouble / target.count))
            .sortBy(_._2)
            .reverse
            .head
            ._2

          val r = (
            dsName,
            Series(
              "majority-class-frequency" -> inbalance,
              "lamp-accuracy" -> lampAccuracy1,
              "lamp-accuracy" -> lampAccuracy2,
              "torch-accuracy" -> torchAccuracy1,
              "lamp-time" -> (t2 - t1) / 1e9,
              "lamp-time" -> (t3 - t2) / 1e9,
              "torch-time" -> torchTime1
            )
          )
          // ???
          r
      }
      .toFrame
      .T
      .sortedRIx

    val lampMean = accuracies.col("lamp-accuracy").T.mean
    val torchMean = accuracies.col("torch-accuracy").T.mean
    val diffMean = (torchMean - lampMean).toVec.mean
    val r2 = math.pow(lampMean.toVec.pearson(torchMean.toVec), 2d)
    println(new String(org.saddle.csv.CsvWriter.writeFrameToArray(accuracies)))
    println(accuracies.stringify(100, 100))
    println(r2)
    println(diffMean)
    assert(diffMean < 0.1)
    assert(r2 > 0.3)

  }
}
