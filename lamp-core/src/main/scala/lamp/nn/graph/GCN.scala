package lamp.nn.graph

import lamp.nn._
import lamp.autograd.{BatchNorm => _, Dropout => _, _}
import aten.ATen
import lamp.Sc
import lamp.STen
import lamp.STenOptions

case class GCN[M <: Module](
    transform: M with Module
) extends GraphModule {

  def state =
    transform.state

  override def forward[S: Sc](
      x: Graph
  ): Graph = {
    val message = GCN.gcnAggregation(x.nodeFeatures, x.edgeI, x.edgeJ)
    val transformedNodes = transform.forward(message)

    x.copy(nodeFeatures = transformedNodes)
  }

}

object GCN {

  def computeSparseAdjacency[S: Sc](
      valueOpt: STenOptions,
      edgeI: STen,
      edgeJ: STen,
      numNodes: Long
  ) = {

    val edgeList = STen.stack(List(edgeI, edgeJ), dim = 1)

    val tOptDoubleDevice =
      if (valueOpt.isCPU)
        STenOptions
          .fromScalarType(valueOpt.scalarTypeByte)
          .cpu
      else
        STenOptions
          .fromScalarType(valueOpt.scalarTypeByte)
          .cudaIndex(valueOpt.deviceIndex.toShort)

    val tOptLongDevice =
      if (valueOpt.isCPU) STenOptions.l.cpu
      else STenOptions.l.cudaIndex(valueOpt.deviceIndex.toShort)

    val degrees = {
      val counts = edgeList
        .view(-1)
        .bincount(
          weights = None,
          minLength = numNodes.toInt
        )
      counts += 1L

      const(counts.pow(-0.5).unsqueeze(1))

    }

    val a = {
      val ones = ATen.ones(Array(edgeList.shape(0)), tOptDoubleDevice.value)
      val edgeListT = ATen.t(edgeList.value)
      val sp1 = ATen.sparse_coo_tensor(
        edgeListT,
        ones,
        Array(numNodes, numNodes),
        tOptDoubleDevice.value
      )

      val sp1T = ATen.t(sp1)
      ATen.add_out(sp1, sp1, sp1T, 1d)
      edgeListT.release
      sp1T.release
      ones.release
      val ident = {
        val selfLoops = {
          val ar =
            ATen.arange_2(0d, numNodes.toDouble, 1.0, tOptLongDevice.value)
          val ar2 = ATen._unsafe_view(ar, Array(-1, 1))
          val r = ar2.repeat(Array(1, 2))
          ar.release
          ar2.release
          val rt = ATen.t(r)
          r.release
          rt
        }
        val ones =
          ATen.ones(Array(selfLoops.sizes.apply(1)), tOptDoubleDevice.value)
        val i = ATen.sparse_coo_tensor(
          selfLoops,
          ones,
          Array(numNodes, numNodes),
          tOptDoubleDevice.value
        )
        ones.release
        selfLoops.release
        i
      }
      ATen.add_out(sp1, sp1, ident, 1d)

      ident.release
      const(STen.owned(sp1))
    }

    (degrees, a)
  }

  /** Performs D^-0.5 (A+A'+I) D^-0.5 W where are node features (N x D) A is the
    * asymmetric adjacency matrix without self loops, elements in {0,1} I is
    * identity D is degree(A+I)
    *
    * @param nodeFeatures
    *   N x D node features
    * @param edgeList
    *   N x 2 long tensor the edges in A (asymmetric, no diagonal)
    * @return
    *   N x D aggregated features
    */
  def gcnAggregation[S: Sc](
      nodeFeatures: Variable,
      edgeI: STen,
      edgeJ: STen
  ): Variable = {
    val (degrees, a) = computeSparseAdjacency(
      nodeFeatures.options,
      edgeI,
      edgeJ,
      nodeFeatures.sizes(0)
    )
    gcnAggregation(nodeFeatures, degrees, a)
  }
  def gcnAggregation[S: Sc](
      nodeFeatures: Variable,
      degrees: Variable,
      a: Variable
  ): Variable = {

    degrees * (a.mm(nodeFeatures * degrees))
  }

  implicit def trainingMode[M <: Module: TrainingMode] : TrainingMode[GCN[M]]=
    TrainingMode
      .make[GCN[M]](
        m => m.copy(transform = m.transform.asEval),
        m => m.copy(transform = m.transform.asTraining)
      )
  implicit def load[M <: Module: Load] : Load[GCN[M]] = Load.make[GCN[M]] { m => tensors =>
    m.transform.load(tensors)

  }

  def gcn[S: Sc](
      in: Int,
      out: Int,
      tOpt: STenOptions,
      dropout: Double = 0d,
      nonLinearity: Boolean = true
  ) =
    GCN(
      ResidualModule(
        EitherModule(
          if (nonLinearity)
            Left(
              sequence(
                Linear(in = in, out = out, tOpt = tOpt, bias = false),
                BatchNorm(features = out, tOpt = tOpt),
                Fun(scope => input => input.relu(scope)),
                Dropout(dropout, training = true)
              )
            )
          else
            Right(
              sequence(
                Linear(in = in, out = out, tOpt = tOpt, bias = false),
                BatchNorm(features = out, tOpt = tOpt)
              )
            )
        )
      )
    )

}
