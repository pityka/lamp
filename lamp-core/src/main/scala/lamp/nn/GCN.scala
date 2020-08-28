package lamp.nn

import lamp.autograd._
import aten.ATen
import aten.TensorOptions
import lamp.CudaDevice
import lamp.CPU

case class GCN(
    weightsFH: Variable,
    bias: Variable,
    dropout: Double,
    train: Boolean,
    relu: Boolean
) extends GenericModule[
      (Variable, Variable),
      (Variable, Variable)
    ] {

  def state: Seq[(Variable, PTag)] =
    List(
      weightsFH -> GCN.WeightsFH,
      bias -> GCN.BiasH
    )

  override def forward(
      x: (Variable, Variable)
  ): (Variable, Variable) = {
    val (nodeFeatures, edgeList) = x
    val numNodes = nodeFeatures.sizes(0)
    val edgeListWithSelfLoops = {
      val tOpt = TensorHelpers.device(nodeFeatures.value) match {
        case CPU             => TensorOptions.l.cpu
        case CudaDevice(idx) => TensorOptions.l.cuda_index(idx)
      }
      val selfLoops = {
        val ar = ATen.arange(0d, numNodes.toDouble, 1.0, tOpt)
        val ar2 = ATen._unsafe_view(ar, Array(-1, 1))
        val r = ar2.repeat(Array(1, 2))
        ar.release
        ar2.release
        const(r)(nodeFeatures.pool).releasable
      }
      edgeList.cat(selfLoops, 0)
    }
    val edgeFrom = edgeListWithSelfLoops.select(1, 0)
    val edgeTo = edgeListWithSelfLoops.select(1, 1)

    val degrees = {
      val ones = ATen.ones(Array(edgeFrom.shape(0)), nodeFeatures.options)
      val zeros = ATen.zeros(Array(numNodes), ones.options)
      val result = ATen.scatter_add(zeros, 0, edgeFrom.value, ones)
      ones.release
      zeros.release
      const(result)(nodeFeatures.pool).releasable.pow(-0.5)
    }
    val scatteredDegreesTo =
      degrees.indexSelect(0, edgeTo)

    val scatteredNodes = (nodeFeatures * degrees.view(List(-1, 1)))
      .indexSelect(0, edgeFrom)

    val weightedScatteredNodes =
      scatteredNodes * scatteredDegreesTo.view(List(-1, 1))
    val message = {
      weightedScatteredNodes.scatterAdd(
        edgeTo.view(List(-1, 1)).expandAs(scatteredNodes.value),
        0,
        numNodes
      )
    }
    val hiddenSize = weightsFH.shape(1)
    val transformedNodes = {
      val x1 = (message.mm(weightsFH) + bias)
      val x2 = if (relu) x1.relu.dropout(dropout, train) else x1

      if (nodeFeatures.shape(1) == hiddenSize)
        x2 + nodeFeatures
      else x2
    }
    (transformedNodes, edgeList)
  }

}

object GCN {
  case object WeightsFH extends LeafTag
  case object BiasH extends LeafTag

  implicit val trainingMode = TrainingMode
    .make[GCN](m => m.copy(train = false), m => m.copy(train = true))
  implicit val load = Load.make[GCN] { m => tensors =>
    implicit val pool = m.weightsFH.pool
    m.copy(
      weightsFH = param(tensors(0)),
      bias = param(tensors(2))
    )
  }

  def apply(
      in: Int,
      hiddenSize: Int,
      tOpt: TensorOptions,
      dropout: Double = 0d,
      relu: Boolean = true
  )(implicit pool: AllocatedVariablePool): GCN =
    GCN(
      weightsFH = param(
        ATen.normal_3(
          0d,
          math.sqrt(2d / (in + hiddenSize)),
          Array(in, hiddenSize),
          tOpt
        )
      ),
      bias = param(
        ATen.zeros(
          Array(1, hiddenSize),
          tOpt
        )
      ),
      dropout = dropout,
      train = true,
      relu = relu
    )
}
