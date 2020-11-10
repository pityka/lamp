package lamp.nn

import lamp.autograd._
import aten.ATen
import aten.TensorOptions
import lamp.Sc
import lamp.scope

case class Passthrough[M <: Module](
    m: M with Module
) extends GenericModule[
      (Variable, Variable),
      (Variable, Variable)
    ] {

  def state: Seq[(Variable, PTag)] =
    m.state

  override def forward[S: Sc](
      x: (Variable, Variable)
  ): (Variable, Variable) = {
    val (a, b) = x

    (m.forward(a), b)
  }

}
object Passthrough {
  implicit def trainingMode[M <: Module: TrainingMode] =
    TrainingMode
      .make[Passthrough[M]](
        m => m.copy(m = m.m.asEval),
        m => m.copy(m = m.m.asTraining)
      )
  implicit def load[M <: Module: Load] = Load.make[Passthrough[M]] {
    m => tensors =>
      m.copy(
        m = m.m.load(tensors)
      )
  }
}
case class GCN[M <: Module](
    transform: M with Module
) extends GenericModule[
      (Variable, Variable),
      (Variable, Variable)
    ] {

  def state: Seq[(Variable, PTag)] =
    transform.state

  override def forward[S: Sc](
      x: (Variable, Variable)
  ): (Variable, Variable) = {
    val (nodeFeatures, edgeList) = x
    val message = GCN.gcnAggregation(nodeFeatures, edgeList)
    val transformedNodes = transform.forward(message)

    (transformedNodes, edgeList)
  }

}

case class NGCN[M <: Module](
    transforms: Seq[M with Module],
    weightFc: Variable,
    K: Int,
    includeZeroOrder: Boolean
) extends GenericModule[
      (Variable, Variable),
      (Variable, Variable)
    ] {

  def state: Seq[(Variable, PTag)] =
    (weightFc -> NGCN.Weights) +:
      transforms.flatMap(_.state)

  override def forward[S: Sc](
      x: (Variable, Variable)
  ): (Variable, Variable) = {
    val (nodeFeatures, edgeList) = x
    val (degrees, a) = GCN.precomputeSparseAdjacency(
      nodeFeatures.options,
      edgeList,
      nodeFeatures.sizes(0)
    )

    val K = transforms.size
    val to = if (includeZeroOrder) K - 1 else K
    val drop = if (includeZeroOrder) 0 else 1
    val messages = (0 until to)
      .scanLeft(nodeFeatures) { (x, _) => GCN.gcnAggregation(x, degrees, a) }
      .drop(drop)
    assert(messages.size == K)
    val transformedNodes = (transforms.grouped(K).toList zip messages) map {
      case (transforms, message) =>
        transforms.map { tr => tr.forward(message) }
    }
    val cat = Concatenate(scope, transformedNodes.flatten, 1).value

    (cat.mm(weightFc), edgeList)
  }

}

object NGCN {

  case object Weights extends LeafTag

  implicit def trainingModeN[M <: Module: TrainingMode]: TrainingMode[NGCN[M]] =
    TrainingMode
      .make[NGCN[M]](
        m => m.copy(transforms = m.transforms.map(_.asEval)),
        m => m.copy(transforms = m.transforms.map(_.asTraining))
      )
  implicit def loadN[M <: Module: Load] = Load.make[NGCN[M]] { m => tensors =>
    val w = tensors.head
    val (loadedMembers, _) =
      m.transforms.foldLeft((List[M](), tensors.tail)) {
        case ((acc, params), member) =>
          val numParam = member.state.size
          val loaded = member.load(params.take(numParam))
          (acc.:+(loaded), params.drop(numParam))

      }
    m.copy(
      transforms = loadedMembers,
      weightFc = param(w)(m.weightFc.pool)
    )
  }

  def ngcn[S: Sc](
      in: Int,
      middle: Int,
      out: Int,
      tOpt: TensorOptions,
      dropout: Double = 0d,
      nonLinearity: Boolean = true,
      K: Int,
      r: Int,
      includeZeroOrder: Boolean = true
  ) = {

    def makeModule = ResidualModule(
      EitherModule(
        if (nonLinearity)
          Left(
            sequence(
              Linear(in = in, out = middle, tOpt = tOpt, bias = false),
              BatchNorm(features = middle, tOpt = tOpt),
              Fun(scope => input => input.relu(scope)),
              Dropout(dropout, training = true)
            )
          )
        else
          Right(Linear(in = in, out = middle, tOpt = tOpt, bias = false))
      )
    )

    NGCN(
      0 until K * r map (_ => makeModule),
      weightFc = param(
        ATen.normal_3(
          0d,
          math.sqrt(2d / (out + middle * r * K)),
          Array(middle * r * K, out),
          tOpt
        )
      ),
      K,
      includeZeroOrder
    )
  }
}

object GCN {

  def precomputeSparseAdjacency[S: Sc](
      valueOpt: TensorOptions,
      edgeList: Variable,
      numNodes: Long
  ) = {

    val edgeFrom = edgeList.select(1, 0)
    val edgeTo = edgeList.select(1, 1)

    val tOptDoubleDevice =
      if (valueOpt.isCPU())
        TensorOptions
          .fromScalarType(valueOpt.scalarTypeByte())
          .cpu()
      else
        TensorOptions
          .fromScalarType(valueOpt.scalarTypeByte())
          .cuda_index(valueOpt.deviceIndex())

    val tOptLongDevice =
      if (valueOpt.isCPU) TensorOptions.l.cpu
      else TensorOptions.l.cuda_index(valueOpt.deviceIndex())

    val degrees = {
      val ones = ATen.ones(Array(edgeFrom.shape(0)), tOptDoubleDevice)
      val zeros = ATen.zeros(Array(numNodes), ones.options)
      val result1 = ATen.index_add(zeros, 0, edgeFrom.value, ones)
      val result2 = ATen.index_add(result1, 0, edgeTo.value, ones)
      val result3 = ATen.add_1(result2, 1.0, 1.0)
      val viewed = ATen._unsafe_view(result3, Array(-1, 1))
      ones.release
      zeros.release
      result1.release
      result2.release
      result3.release
      const(viewed).pow(-0.5)
    }

    val a = {
      val ones = ATen.ones(Array(edgeList.shape(0)), tOptDoubleDevice)
      val edgeListT = ATen.t(edgeList.value)
      val sp1 = ATen.sparse_coo_tensor(
        edgeListT,
        ones,
        Array(numNodes, numNodes),
        tOptDoubleDevice
      )

      val sp1T = ATen.t(sp1)
      ATen.add_out(sp1, sp1, sp1T, 1d)
      edgeListT.release
      sp1T.release
      ones.release
      val ident = {
        val selfLoops = {
          val ar = ATen.arange(0d, numNodes.toDouble, 1.0, tOptLongDevice)
          val ar2 = ATen._unsafe_view(ar, Array(-1, 1))
          val r = ar2.repeat(Array(1, 2))
          ar.release
          ar2.release
          val rt = ATen.t(r)
          r.release
          rt
        }
        val ones = ATen.ones(Array(selfLoops.sizes.apply(1)), tOptDoubleDevice)
        val i = ATen.sparse_coo_tensor(
          selfLoops,
          ones,
          Array(numNodes, numNodes),
          tOptDoubleDevice
        )
        ones.release
        selfLoops.release
        i
      }
      ATen.add_out(sp1, sp1, ident, 1d)

      ident.release
      const(sp1)
    }

    (degrees, a)
  }

  /**
    * Performs D^-0.5 (A+A'+I) D^-0.5 W
    *   where are node features (N x D)
    *   A is the asymmetric adjacency matrix without self loops, elements in {0,1}
    *   I is identity
    *   D is degree(A+I)
    *
    *
    * @param nodeFeatures N x D node features
    * @param edgeList N x 2 long tensor the edges in A (asymmetric, no diagonal)
    * @return N x D aggregated features
    */
  def gcnAggregation[S: Sc](
      nodeFeatures: Variable,
      edgeList: Variable
  ): Variable = {
    val (degrees, a) = precomputeSparseAdjacency(
      nodeFeatures.options,
      edgeList,
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

  implicit def trainingMode[M <: Module: TrainingMode] =
    TrainingMode
      .make[GCN[M]](
        m => m.copy(transform = m.transform.asEval),
        m => m.copy(transform = m.transform.asTraining)
      )
  implicit def load[M <: Module: Load] = Load.make[GCN[M]] { m => tensors =>
    m.copy(
      transform = m.transform.load(tensors)
    )
  }

  def gcn[S: Sc](
      in: Int,
      out: Int,
      tOpt: TensorOptions,
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
