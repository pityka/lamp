package lamp.umap

import lamp._
import org.saddle._
import org.saddle.linalg._
import org.saddle.macros.BinOps._
import lamp.autograd._
import lamp.nn.AdamW
import lamp.nn.simple
import lamp.nn.NoTag
import scribe.Logger
import aten.ATen
import aten.Tensor

object Umap {

  private[lamp] def binarySearch(
      target: Double,
      min: Double,
      max: Double,
      mid: Double,
      it: Int,
      eps: Double
  )(
      fun: Double => Double
  ): Double = {
    if (it > 1000) mid
    else {
      val atMid = fun(mid)
      if (math.abs(atMid - target) < eps) mid
      else if (atMid > target) {
        binarySearch(
          target = target,
          min = min,
          max = mid,
          mid = (min + mid) * 0.5,
          it = it + 1,
          eps = eps
        )(fun)
      } else {
        binarySearch(
          target = target,
          min = mid,
          max = max,
          mid = if (max.isPosInfinity) mid * 2 else (max + mid) * 0.5,
          it = it + 1,
          eps = eps
        )(fun)
      }
    }
  }

  private[lamp] def edgeWeights(
      data: Mat[Double],
      knn: Mat[Int]
  ): Mat[Double] = {
    val knnDistances = knn.mapRows {
      case (row, rowIdx) =>
        val row1 = data.row(rowIdx)
        row.map { idx2 =>
          val row2 = data.row(idx2)
          val d = row1 - row2
          math.sqrt(d vv d)
        }
    }
    val rho = array
      .range(0, data.numRows)
      .map { rowIdx => knnDistances.row(rowIdx).filter(_ > 0d).min2 }
      .toVec

    val log2k = math.log(knn.numCols) / math.log(2d)
    val sigma = array
      .range(0, data.numRows)
      .map { rowIdx =>
        val r = rho.raw(rowIdx)
        def fun(s: Double) =
          knnDistances
            .row(rowIdx)
            .map(d => math.exp((-1 * math.max(0, d - r)) / s))
            .sum
        binarySearch(
          target = log2k,
          min = 0d,
          max = Double.PositiveInfinity,
          mid = 1d,
          it = 0,
          eps = 1e-6
        )(fun)
      }
      .toVec

    val b = Mat(
      array
        .range(0, data.numRows)
        .flatMap { i =>
          val r = rho.raw(i)
          val s = sigma.raw(i)
          val dists = knnDistances.row(i)
          knn.row(i).toSeq.zipWithIndex.flatMap {
            case (j, jidx) =>
              if (i == j) Nil
              else {
                val d = dists.raw(jidx)
                val wij = math.exp((-1 * math.max(0, d - r)) / s)
                val wji = {
                  val r = rho.raw(j)
                  val s = sigma.raw(j)
                  val l = knn.row(j).findOne(_ == i)
                  if (l == -1) 0d
                  else {
                    val d = knnDistances.raw(j, l)
                    math.exp((-1 * math.max(0, d - r)) / s)
                  }
                }
                val b = wij + wji - wij * wji
                List(Vec(i.toDouble, j.toDouble, b))
              }
          }
        }: _*
    ).T

    b

  }

  private[lamp] def optimize(
      edgeWeights: Mat[Double],
      total: Int,
      lr: Double,
      iterations: Int,
      minDist: Double,
      negativeSampleSize: Int,
      randomSeed: Long,
      balanceAttractionsAndRepulsions: Boolean,
      repulsionStrength: Double,
      logger: Option[Logger],
      device: Device,
      numDim: Int
  ) = {
    val precision = DoublePrecision

    def loss(
        locations: Variable,
        index1: Variable,
        index2: Variable,
        index3: Variable,
        index4: Variable,
        b: Variable
    ) = {

      val locations_1 = locations.indexSelect(0, index1)
      val locations_2 = locations.indexSelect(0, index2)
      val locations_3 = locations.indexSelect(0, index3)
      val locations_4 = locations.indexSelect(0, index4)
      val locNorm1 =
        locations_1.euclideanDistance(locations_2, 1).view(List(-1))
      val attractions =
        if (minDist == 0d) {
          (locNorm1 * b).sum * (-1)
        } else {
          (CappedShiftedNegativeExponential(locNorm1, minDist).value.log * b).sum
        }

      val locNorm2 =
        locations_3.euclideanDistance(locations_4, 1).view(List(-1))
      val repulsions =
        if (minDist == 0d) (((locNorm2 * (-1)).exp * (-1))).log1p.sum
        else {
          val p =
            CappedShiftedNegativeExponential(locNorm2, minDist).value * (-1) + 1e-6
          p.log1p.sum
        }

      if (balanceAttractionsAndRepulsions)
        (attractions / b.sum + repulsions * (repulsionStrength / locations_3
          .sizes(0))) * (-1)
      else (attractions + repulsions) * (-1)
    }

    val indexI = edgeWeights.col(0).map(_.toLong)
    val indexIT = TensorHelpers.fromLongVec(indexI, device)
    val rng = org.saddle.spire.random.rng.Cmwc5.fromTime(randomSeed)
    Tensor.manual_seed(randomSeed)

    def sampleRepulsivePairsT(n: Int) = {
      val ii = ATen.repeat_interleave_2(indexIT, n, 0)

      val m = ii.sizes.apply(0)
      val jj = ATen.randint_2(0, total - 1, Array(m), ii.options)
      val mask = ATen.ne_1(ii, jj)
      val ri = ATen.masked_select(ii, mask)
      val rj = ATen.masked_select(jj, mask)
      ii.release
      jj.release
      mask.release
      (ri, rj)
    }

    implicit val pool = new AllocatedVariablePool
    val locations = param(
      TensorHelpers.fromMat(
        Mat(total, numDim, array.randDouble(total * numDim, rng)),
        device,
        precision
      )
    )
    val index1 = const(
      indexIT
    )
    val index2 = {
      val indexJ = edgeWeights.col(1).map(_.toLong)
      const(
        TensorHelpers.fromLongVec(indexJ, device)
      )
    }

    val b = const(
      TensorHelpers.fromVec(edgeWeights.col(2), device, precision)
    )

    val optimizer = AdamW.factory(
      weightDecay = simple(0.0),
      learningRate = simple(lr),
      clip = Some(1d)
    )(List(locations.value -> NoTag))

    var i = 0
    var lastLoss = 0d
    while (i < iterations) {

      val (index3T, index4T) = {
        var (index3, index4) = sampleRepulsivePairsT(negativeSampleSize)
        val i3 = const(
          index3
        ).releasable
        val i4 = const(
          index4
        ).releasable
        (i3, i4)
      }

      val lossV = loss(
        locations = locations,
        index1 = index1,
        index2 = index2,
        index3 = index3T,
        index4 = index4T,
        b = b
      )
      val lossAsDouble = lossV.value.toMat.raw(0)
      lastLoss = lossAsDouble
      logger.foreach(_.info(s"loss in epoch: ${(i, lossAsDouble)}"))

      val gradients = {
        locations.zeroGrad()
        lossV.backprop()
        val g = locations.partialDerivative
        lossV.releaseAll
        g
      }
      optimizer.step(List(gradients))
      i += 1
    }
    pool.releaseAll()
    index1.value.release()
    index2.value.release()
    b.value.release()
    optimizer.release()

    val jLoc = locations.toMat
    locations.value.release
    (jLoc, lastLoss)

  }

  /**
    * Dimension reduction similar to UMAP
    * For reference see https://arxiv.org/abs/1802.03426
    * This method does not follow the above paper exactly.
    *
    * Minimizes the objective function:
    * L(x) = L_attraction(x) + L_repulsion(x)
    *
    * L_attraction(x) = sum over (i,j) edges : b_ij * ln(f(x_i,x_j))
    * b_ij is the value of the 'UMAP graph' as in the above paper
    * x_i is the low dimensional coordinate of the i-th sample
    * f(x,y) = 1 if ||x-y||_2 < minDist , or exp(-(||x-y||_2 - minDist)) otherwise
    *
    * L_repulsion(x) = sum over (i,j) edges: (1-b_ij) * ln(1 - f(x_i,x_j)) , evaluated with sampling
    * L_repulsion is evaluated by randomly sampling in each iteration from all (i,j) edges having b_ij=0
    *
    * Nearest neighbor search is evaluated by brute force.
    * It may be batched, and may be evaluated on the GPU.
    *
    * L(x) is maximized by gradient descent, in particular Adam.
    * Derivatives of L(x) are computed using reverse mode automatic differentiation (autograd).
    * Gradient descent may be evaluated on the GPU.
    *
    * Distance metric is alway Euclidean.
    *
    * Differences to the algorithm described in the UMAP paper:
    *   - The paper desribes a smooth approximation of the function 'f' (Definition 11.).
    *     That approximation is not used in this code.
    *   - The paper describes an optimization procedure different from the approach taken here.
    *     They sample each edge according to b_ij and update the vertices one after the other.
    *     The current code updates each locations all together according to the derivative of L(x).
    *
    * @param data each row is a sample
    * @param device device to run the optimization and KNN search (GPU or CPU)
    * @param precision precision to run the KNN search, optimization is always in double precision
    * @param k number of nearest neighbors to retrieve. Self is counted as nearest neighbor
    * @param numDim number of dimensions to project to
    * @param knnMinibatchSize KNN search may be batched if the device can't fit the whole distance matrix
    * @param lr learning rate
    * @param iterations number of epochs of optimization
    * @param minDist see above equations for the definition, see the UMAP paper for its effect
    * @param negativeSampleSize number of negative edges to select for each positive
    * @param randomSeed
    * @param balanceAttractionsAndRepulsions if true the number of negative samples will not affect the relative strength of attractions and repulsions (see @param repulsionStrength)
    * @param repulsionStrength strength of repulsions compared to attractions
    * @param logger
    * @return
    */
  def umap(
      data: Mat[Double],
      device: Device = CPU,
      precision: FloatingPointPrecision = DoublePrecision,
      k: Int = 10,
      numDim: Int = 2,
      knnMinibatchSize: Int = 1000,
      lr: Double = 0.1,
      iterations: Int = 500,
      minDist: Double = 0.0d,
      negativeSampleSize: Int = 5,
      randomSeed: Long = 42L,
      balanceAttractionsAndRepulsions: Boolean = true,
      repulsionStrength: Double = 1d,
      logger: Option[Logger] = None
  ) = {
    val knn = lamp.knn.knnSearch(
      data,
      data,
      k,
      lamp.knn.squaredEuclideanDistance _,
      device,
      precision,
      knnMinibatchSize
    )

    logger.foreach(_.info("KNN done"))

    val b = edgeWeights(data, knn)

    logger.foreach(_.info(s"${b.numRows} edge weights computed"))

    val (layout, loss) =
      optimize(
        b,
        data.numRows,
        lr = lr,
        iterations = iterations,
        minDist = minDist,
        negativeSampleSize = negativeSampleSize,
        randomSeed = randomSeed,
        balanceAttractionsAndRepulsions = balanceAttractionsAndRepulsions,
        repulsionStrength = repulsionStrength,
        logger = logger,
        device = device,
        numDim = numDim
      )

    logger.foreach(_.info(s"optimization done"))

    (layout, b, loss)
  }

}