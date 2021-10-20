package lamp.tgnn

import lamp._
import lamp.autograd._
import lamp.nn._
import lamp.nn.graph.Graph

sealed trait ColumnDataType
case class VariableColumn(variable: Variable) extends ColumnDataType

sealed trait ColumnModule extends GenericModule[ColumnDataType, Variable]

object ColumnModule {

  implicit val trainingMode: TrainingMode[ColumnModule] =
    TrainingMode
      .make[ColumnModule](
        _ match {
          case ColumnEmbeddingModule(m2) => ColumnEmbeddingModule(m2.asEval)
          case ColumnLinearModule(m2)    => ColumnLinearModule(m2.asEval)
        },
        _ match {
          case ColumnEmbeddingModule(m2) => ColumnEmbeddingModule(m2.asTraining)
          case ColumnLinearModule(m2)    => ColumnLinearModule(m2.asTraining)
        }
      )

  implicit val load: Load[ColumnModule] = Load.make[ColumnModule] {
    m => tensors =>
      m match {
        case ColumnEmbeddingModule(embedding) => embedding.load(tensors)
        case ColumnLinearModule(embedding)    => embedding.load(tensors)
      }
  }

}

case class ColumnEmbeddingModule(embedding: lamp.nn.Embedding)
    extends ColumnModule {

  override def forward[S: Sc](x: ColumnDataType): Variable = x match {
    case VariableColumn(variable) => embedding.forward(variable)
    case _                        => ???
  }

  override def state: Seq[(Constant, PTag)] = embedding.state
}
case class ColumnLinearModule(embedding: lamp.nn.Linear) extends ColumnModule {

  override def forward[S: Sc](x: ColumnDataType): Variable = x match {
    case VariableColumn(variable) => embedding.forward(variable)
    case _                        => ???
  }

  override def state: Seq[(Constant, PTag)] = embedding.state

}

case class AttributeColumns(
    attributeColumns: Seq[ColumnDataType],
    sampleIdxOfAttributes: STen
)

case class Batch(
    sampleColumns: Seq[ColumnDataType],
    attributes: Seq[AttributeColumns]
)

case class EmbeddedAttribute(
    attributes: Variable,
    sampleIdxOfAttributes: STen
)

case class SampleAttributeGNN[M <: GenericModule[Graph, Graph]](
    sampleEmbedder: SampleEmbedder,
    attributeEmbedders: Seq[AttributeEmbedder],
    graphNN: M with GenericModule[Graph, Graph]
) extends GenericModule[Batch, Variable] {

  override def forward[S: Sc](x: Batch): Variable = {
    val embeddedSample = sampleEmbedder.forward(x.sampleColumns)
    val embeddedAttributes = attributeEmbedders.zip(x.attributes).map {
      case (embedder, attribute) => embedder.forward(attribute)
    }
    val graph = SampleAttributeGNN.toGraph(embeddedSample, embeddedAttributes)
    val updatedGraph = graphNN.forward(graph)
    val updatedSamples = updatedGraph.nodeFeatures.indexSelect(
      dim = 0,
      const(updatedGraph.vertexPoolingIndices)
    )
    updatedSamples
  }

  override def state: Seq[(Constant, PTag)] =
    sampleEmbedder.state ++ graphNN.state ++ attributeEmbedders.flatMap(_.state)

}

object SampleAttributeGNN {

  def make[S: Sc](
      sampleColumnTypes: Seq[EmbeddingType],
      attributesColumnTypes: Seq[Seq[EmbeddingType]],
      nodeDimension: Int,
      dropout: Double,
      tOpt: STenOptions
  ) = {

    SampleAttributeGNN(
      sampleEmbedder =
        SampleEmbedder.apply(columnEmbeddingTypes = sampleColumnTypes, nodeDimension,tOpt),
      attributeEmbedders = attributesColumnTypes.map(attributes =>
        AttributeEmbedder.apply(attributes, nodeDimension,tOpt)
      ),
      graphNN = Sequential(
        lamp.nn.graph.MPNN(
          messageTransform = lamp.nn.MLP(
            in = nodeDimension * 2 + 1,
            out = nodeDimension,
            hidden = List(nodeDimension),
            tOpt = tOpt,
            dropout = dropout,
            lastNonLinearity = true,
            activationFunction = MLP.Relu,
            norm = MLP.NormType.LayerNorm
          ),
          vertexTransform = lamp.nn.MLP(
            in = nodeDimension * 2,
            out = nodeDimension,
            hidden = List(nodeDimension),
            tOpt = tOpt,
            dropout = dropout,
            lastNonLinearity = true,
            activationFunction = MLP.Relu,
            norm = MLP.NormType.LayerNorm
          ),
          degreeNormalizeI = true,
          degreeNormalizeJ = false,
          aggregateJ = false
        )
      )
    )
  }

  implicit def trainingMode[M <: GenericModule[Graph, Graph]: TrainingMode]
      : TrainingMode[SampleAttributeGNN[M]] =
    TrainingMode
      .make[SampleAttributeGNN[M]](
        m =>
          m.copy(
            sampleEmbedder = m.sampleEmbedder.asEval,
            attributeEmbedders = m.attributeEmbedders.map(_.asEval),
            graphNN = m.graphNN.asEval
          ),
        m =>
          m.copy(
            sampleEmbedder = m.sampleEmbedder.asTraining,
            attributeEmbedders = m.attributeEmbedders.map(_.asTraining),
            graphNN = m.graphNN.asTraining
          )
      )
  implicit def load[M <: GenericModule[Graph, Graph]: Load]
      : Load[SampleAttributeGNN[M]] =
    Load.make[SampleAttributeGNN[M]] { m => tensors =>
      m.sampleEmbedder.load(tensors.take(m.sampleEmbedder.state.size))
      m.graphNN.load(
        tensors.drop(m.sampleEmbedder.state.size).take(m.graphNN.state.size)
      )

      m.attributeEmbedders.foldLeft(
        tensors.drop(m.sampleEmbedder.state.size + m.graphNN.state.size)
      ) { (tensors, module) =>
        val sze = module.state.size
        module.load(tensors.take(sze))
        tensors.drop(sze)
      }

    }

  def toGraph[S: Sc](
      samples: Variable,
      attributes: Seq[EmbeddedAttribute]
  ): Graph = {
    val edgeI = STen.cat(attributes.map(_.sampleIdxOfAttributes), dim = 0)
    val edgeJ = STen.arange_l(
      0L,
      attributes.map(_.attributes.shape(0)).foldLeft(0L)(_ + _),
      1L,
      edgeI.options
    ) + samples.shape(0)

    val nodes =
      Variable.cat(List(samples) ++ attributes.map(_.attributes), dim = 0)
    val edgeFeatures = const(STen.randn(List(edgeI.shape(0))))

    val sampleIdx = STen.arange_l(0L, samples.shape(0), 1L, edgeI.options)

    val vertexPooling = sampleIdx //STen.cat(List(edgeI, sampleIdx),dim=0)

    lamp.nn.graph.Graph(
      nodeFeatures = nodes,
      edgeFeatures = edgeFeatures,
      edgeI = edgeI,
      edgeJ = edgeJ,
      vertexPoolingIndices = vertexPooling
    )
  }
}

sealed trait EmbeddingType
case class Embedder(numClasses: Int, outDim: Int) extends EmbeddingType
case class Linear(inDim: Int, outDim: Int) extends EmbeddingType
// case class NLP(length: Int, outDim: Int) extends EmbeddingType

case class AttributeEmbedder(
    embedders: Seq[ColumnModule],
    linear: lamp.nn.Linear
) extends GenericModule[AttributeColumns, EmbeddedAttribute] {
  def state = linear.state ++ embedders.flatMap(_.state)
  def forward[S: Sc](x: AttributeColumns): EmbeddedAttribute = {
    val embeddedColumns = embedders.zip(x.attributeColumns).map {
      case (embedder, column) => embedder.forward(column)
    }
    val cat = linear.forward(Variable.cat(embeddedColumns, dim = 1))
    EmbeddedAttribute(
      attributes = cat,
      sampleIdxOfAttributes = x.sampleIdxOfAttributes
    )
  }
}

object AttributeEmbedder {

  implicit val trainingMode: TrainingMode[AttributeEmbedder] =
    TrainingMode
      .make[AttributeEmbedder](
        m =>
          m.copy(
            embedders = m.embedders.map(_.asEval),
            linear = m.linear.asEval
          ),
        m =>
          m.copy(
            embedders = m.embedders.map(_.asTraining),
            linear = m.linear.asTraining
          )
      )
  implicit val load: Load[AttributeEmbedder] = Load.make[AttributeEmbedder] {
    m => tensors =>
      m.linear.load(tensors.take(m.linear.state.size))
      m.embedders.foldLeft(tensors.drop(m.linear.state.size)) {
        (tensors, module) =>
          val sze = module.state.size
          module.load(tensors.take(sze))
          tensors.drop(sze)
      }
  }

  def apply[S: Sc](
      columnEmbeddingTypes: Seq[EmbeddingType],
      outDim: Int,
      tOpt: STenOptions
  ): AttributeEmbedder = {
    val d = columnEmbeddingTypes.map {
      _ match {
        case Embedder(_, outDim) => outDim
        case Linear(_, outDim)   => outDim
      }
    }.sum
    AttributeEmbedder(
      columnEmbeddingTypes map { attr =>
        attr match {
          case Embedder(numClasses, outDim) =>
            ColumnEmbeddingModule(
              lamp.nn
                .Embedding(
                  classes = numClasses,
                  dimensions = outDim,
                  tOpt = tOpt
                )
            )
          case Linear(in, out) =>
            ColumnLinearModule(
              lamp.nn.Linear(in = in, out = out, tOpt = tOpt, bias = true)
            )
        }
      },
      lamp.nn.Linear(in = d, out = outDim, tOpt, bias = true)
    )
  }
}

case class SampleEmbedder(
    embedders: Seq[ColumnModule],
    linear: lamp.nn.Linear
) extends GenericModule[Seq[ColumnDataType], Variable] {
  def state = linear.state ++ embedders.flatMap(_.state)
  def forward[S: Sc](x: Seq[ColumnDataType]): Variable = {
    val embeddedColumns = embedders.zip(x).map { case (embedder, column) =>
      embedder.forward(column)
    }
    val cat = Variable.cat(embeddedColumns, dim = 1)
    linear.forward(cat)
  }
}

object SampleEmbedder {

  implicit val trainingMode: TrainingMode[SampleEmbedder] =
    TrainingMode
      .make[SampleEmbedder](
        m =>
          m.copy(
            embedders = m.embedders.map(_.asEval),
            linear = m.linear.asEval
          ),
        m =>
          m.copy(
            embedders = m.embedders.map(_.asTraining),
            linear = m.linear.asTraining
          )
      )
  implicit val load: Load[SampleEmbedder] = Load.make[SampleEmbedder] {
    m => tensors =>
      m.linear.load(tensors.take(m.linear.state.size))
      m.embedders.foldLeft(tensors.drop(m.linear.state.size)) {
        (tensors, module) =>
          val sze = module.state.size
          module.load(tensors.take(sze))
          tensors.drop(sze)
      }
  }
  def apply[S: Sc](
      columnEmbeddingTypes: Seq[EmbeddingType],
      outputDim: Int,
      tOpt: STenOptions
  ): SampleEmbedder = {
    val d = columnEmbeddingTypes.map {
      _ match {
        case Embedder(_, outDim) => outDim
        case Linear(_, outDim)   => outDim
      }
    }.sum
    SampleEmbedder(
      columnEmbeddingTypes map {
        _ match {
          case Embedder(numClasses, outDim) =>
            ColumnEmbeddingModule(
              lamp.nn
                .Embedding(
                  classes = numClasses,
                  dimensions = outDim,
                  tOpt = tOpt
                )
            )
          case Linear(in, out) =>
            ColumnLinearModule(
              lamp.nn.Linear(in = in, out = out, tOpt = tOpt, bias = true)
            )
        }
      },
      lamp.nn.Linear(in = d, outputDim, tOpt, true)
    )
  }
}
