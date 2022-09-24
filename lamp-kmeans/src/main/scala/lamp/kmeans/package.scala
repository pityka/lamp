package lamp

package object kmeans {

  /** Minibatch K-Means
    *
    * clusters := clusters * (1-eps) + update(clusters) * eps where update(.) is
    * the Loyd update on a subset of the samples
    *
    * @param instances
    *   Input data of shape [samples,channels]
    * @param clusters
    *   Number of clusters
    * @param iterations
    *   Number of iterations
    * @param learningRate
    *   Learning rate (eps)
    * @param minibatchSize
    *   Size of minibatch
    * @return
    *   triple of (centers, memberships, distance to closest center)
    */
  def minibatchKMeans(
      instances: STen,
      clusters: Int,
      iterations: Int,
      learningRate: Double,
      minibatchSize: Int
  )(implicit scope: Scope): (STen, STen, STen) = {

    def loop(centers: STen, it: Int): STen = if (it == 0) centers
    else {
      Scope { implicit scope =>
        val mb = selectRandomInstancesWithReplacement(instances, minibatchSize)
        val newMeans = findMeansOfClusters(mb, centers)

        val mask = {
          val bools = newMeans.sum(dim = 1, keepDim = false).isnan.unsqueeze(1)
          if (mb.isFloat) bools.castToFloat else bools.castToDouble
        }

        val updated =
          newMeans.nanToNum * learningRate + centers * (1 - learningRate) + centers * mask * learningRate

        loop(updated, it - 1)
      }
    }

    val init: STen = Scope { implicit scope =>
      kmeansPlusPlus(instances, clusters)
    }

    val finalClusterCenters = loop(centers = init, it = iterations)
    val distanceToCenters =
      lamp.knn.squaredEuclideanDistance(instances, finalClusterCenters)
    val min = distanceToCenters.topk(1, 1, false, false)._2
    val distanceToClosestCenters =
      distanceToCenters.gather(index = min, dim = 1).sqrt
    (finalClusterCenters, min, distanceToClosestCenters)

  }

  def kmeansPlusPlusExtendTo(instances: STen, centers: STen, max: Int)(implicit
      scope: Scope
  ) = {

    def extend(cs: STen) = Scope { implicit scope =>
      val d =
        lamp.knn.squaredEuclideanDistance(instances, cs)
      val min = d.topk(1, 1, false, false)._2
      val w = d.gather(index = min, dim = 1).squeeze
      val i = STen.multinomial(w, 1, false)
      cs.cat(instances.indexSelect(dim = 0, index = i), dim = 0)
    }

    val r = (centers.sizes(0) until max).foldLeft(centers)((c, _) => extend(c))
    r

  }
  def kmeansPlusPlus(instances: STen, centers: Int)(implicit scope: Scope) = {

    val i = STen.randint(0, instances.sizes(0), List(1), STenOptions.l)
    val init = instances.indexSelect(0, i)

    kmeansPlusPlusExtendTo(instances, init, centers)
  }

  def findMeansOfClusters(instances: STen, centers: STen)(implicit
      scope: Scope
  ): STen = {
    val distanceToCenters =
      lamp.knn.squaredEuclideanDistance(instances, centers)
    val min = distanceToCenters.topk(1, 1, false, false)._2
    val numCenters = centers.sizes(0).toInt
    val newCenters: STen = STen.stack(
      (0 until numCenters).map { center =>
        val closest = min.squeeze.equ(center.toLong).where.head.squeeze
        instances
          .indexSelect(dim = 0, index = closest)
          .mean(dim = 0, keepDim = false)
      },
      dim = 0
    )
    newCenters
  }

  def selectRandomInstancesWithoutReplacement(instances: STen, num: Int)(
      implicit scope: Scope
  ) = {
    val permuted = STen.randperm(instances.sizes(0), STenOptions.l)
    val idx =
      permuted.slice(dim = 0, start = 0, end = num.toLong, step = 1).view(-1L)
    instances.indexSelect(dim = 0, index = idx)
  }
  def selectRandomInstancesWithReplacement(instances: STen, num: Int)(implicit
      scope: Scope
  ) = {
    val idx = STen.randint(
      low = 0L,
      high = instances.sizes(0),
      size = List(num.toLong),
      tensorOptions = STenOptions.l
    )
    instances.indexSelect(dim = 0, index = idx)
  }

}
