package lamp.tgnn

import lamp._
import lamp.autograd._
import lamp.nn._
import lamp.nn.graph.Graph
import lamp.nn.graph.VertexPooling
import org.saddle._
import org.saddle.index.InnerJoin
import lamp.data.BatchStream
import lamp.data.StreamControl

sealed trait ColumnDataType
case class SingleVariableColumn(variable: Variable) extends ColumnDataType

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
    case SingleVariableColumn(variable) => embedding.forward(variable)
    case _                              => ???
  }

  override def state: Seq[(Constant, PTag)] = embedding.state
}
case class ColumnLinearModule(embedding: lamp.nn.Linear) extends ColumnModule {

  override def forward[S: Sc](x: ColumnDataType): Variable = x match {
    case SingleVariableColumn(variable) => embedding.forward(variable)
    case _                              => ???
  }

  override def state: Seq[(Constant, PTag)] = embedding.state

}

case class TableInColumns(
    columns: Seq[ColumnDataType]
)

case class Batch(
    tables: Seq[TableInColumns],
    edgeI: STen,
    edgeJ: STen,
    vertexPooling: STen
)

case class Relation(
    table1Idx: Int,
    column1Idx: Int,
    table2Idx: Int,
    column2Idx: Int
)

case class Tables(
    dataTables: Seq[TableInColumns],
    keyColumns: IndexedSeq[IndexedSeq[Index[Int]]],
    relations: IndexedSeq[Relation],
    poolMask: IndexedSeq[Vec[Boolean]],
    targets: IndexedSeq[STen]
)
object Batch {

  def subsetBySubject(
      minibatchSize: Int,
      numSubjects: Int,
      rng: Option[org.saddle.spire.random.Generator]
  )(
      subset: (Scope, Device, Vec[Int]) => (Tables, STen)
  ): BatchStream[Batch, Int] = {
    def makeNonEmptyBatch(idx: Array[Int], device: Device) = {
      Scope.inResource.map { implicit scope =>
        val (tables, target) = subset(scope, device, idx.toVec)
        StreamControl(
          (Batch.makeBatch(tables, target.options), target)
        )
      }
    }

    val idx =
      rng
        .map(rng =>
          array
            .shuffle(array.range(0, numSubjects), rng)
            .grouped(minibatchSize)
            .toList
        )
        .getOrElse(
          array
            .range(0, numSubjects)
            .grouped(minibatchSize)
            .toList
        )

    BatchStream.fromIndices(idx.toArray)(makeNonEmptyBatch)

  }

  def makeBatch[S: Sc](
      tables: Tables,
      tOpt: STenOptions
  ) = {
    import tables._
    val device = Device.fromOptions(tOpt)
    val tableLengths = keyColumns.map(_.length)
    val tableOffsets = tableLengths.scanLeft(0L)(_ + _)
    val (is, js) = relations.map { relation =>
      val key1 = keyColumns(relation.table1Idx)(relation.column1Idx)
      val key2 = keyColumns(relation.table2Idx)(relation.column2Idx)
      val intersect = key1.join(key2, InnerJoin)
      val offset1 = intersect.lTake
        .map(ar => STen.fromLongVec(ar.toVec.map(_.toLong), device))
        .getOrElse(
          STen.arange_l(0, key1.length.toLong, 1L, tOpt)
        ) + tableOffsets(relation.table1Idx)
      val offset2 = intersect.rTake
        .map(ar => STen.fromLongVec(ar.toVec.map(_.toLong), device))
        .getOrElse(
          STen.arange_l(0, key2.length.toLong, 1L, tOpt)
        ) + tableOffsets(relation.table2Idx)

      (offset1, offset2)

    }.unzip
    val i = STen.cat(is, dim = 0)
    val j = STen.cat(js, dim = 0)
    val vertexPooling = STen.cat(
      poolMask.zipWithIndex.map { case (mask, tableIdx) =>
        STen
          .fromLongVec(mask.map(v => if (v) 1L else 0L), device)
          .where
          .head + tableOffsets(tableIdx)
      },
      dim = 0
    )
    Batch(dataTables, i, j, vertexPooling)
  }
}

case class EmbeddedTable(
    table: Variable
)

case class TableGNN[M <: GenericModule[Graph, Graph]](
    tableEmbedders: Seq[TableEmbedder],
    graphNN: M with GenericModule[Graph, Graph]
) extends GenericModule[Batch, Variable] {

  override def forward[S: Sc](x: Batch): Variable = {
    val embeddedTables = tableEmbedders.zip(x.tables).map {
      case (embedder, table) => embedder.forward(table)
    }
    val graph =
      TableGNN.toGraph(embeddedTables, x.edgeI, x.edgeJ, x.vertexPooling)
    val updatedGraph = graphNN.forward(graph)
    val pooled = VertexPooling.apply(updatedGraph, VertexPooling.Mean)
    pooled
  }

  override def state: Seq[(Constant, PTag)] =
    graphNN.state ++ tableEmbedders.flatMap(_.state)

}

object TableGNN {

  def make[S: Sc](
      tableColumnTypes: Seq[Seq[EmbeddingType]],
      nodeDimension: Int,
      dropout: Double,
      tOpt: STenOptions
  ) = {

    TableGNN(
      tableEmbedders = tableColumnTypes.map(attributes =>
        TableEmbedder.apply(attributes, nodeDimension, tOpt)
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
      : TrainingMode[TableGNN[M]] =
    TrainingMode
      .make[TableGNN[M]](
        m =>
          m.copy(
            tableEmbedders = m.tableEmbedders.map(_.asEval),
            graphNN = m.graphNN.asEval
          ),
        m =>
          m.copy(
            tableEmbedders = m.tableEmbedders.map(_.asTraining),
            graphNN = m.graphNN.asTraining
          )
      )
  implicit def load[M <: GenericModule[Graph, Graph]: Load]: Load[TableGNN[M]] =
    Load.make[TableGNN[M]] { m => tensors =>
      m.graphNN.load(
        tensors.take(m.graphNN.state.size)
      )

      m.tableEmbedders.foldLeft(
        tensors.drop(m.graphNN.state.size)
      ) { (tensors, module) =>
        val sze = module.state.size
        module.load(tensors.take(sze))
        tensors.drop(sze)
      }

    }

  def toGraph[S: Sc](
      attributes: Seq[EmbeddedTable],
      edgeI: STen,
      edgeJ: STen,
      vertexPooling: STen
  ): Graph = {

    val nodes =
      Variable.cat(attributes.map(_.table), dim = 0)
    val edgeFeatures = const(STen.zeros(List(edgeI.shape(0))))

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

case class TableEmbedder(
    embedders: Seq[ColumnModule],
    linear: lamp.nn.Linear
) extends GenericModule[TableInColumns, EmbeddedTable] {
  def state = linear.state ++ embedders.flatMap(_.state)
  def forward[S: Sc](x: TableInColumns): EmbeddedTable = {
    val embeddedColumns = embedders.zip(x.columns).map {
      case (embedder, column) => embedder.forward(column)
    }
    val cat = linear.forward(Variable.cat(embeddedColumns, dim = 1))
    EmbeddedTable(
      table = cat
    )
  }
}

object TableEmbedder {

  implicit val trainingMode: TrainingMode[TableEmbedder] =
    TrainingMode
      .make[TableEmbedder](
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
  implicit val load: Load[TableEmbedder] = Load.make[TableEmbedder] {
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
  ): TableEmbedder = {
    val d = columnEmbeddingTypes.map {
      _ match {
        case Embedder(_, outDim) => outDim
        case Linear(_, outDim)   => outDim
      }
    }.sum
    TableEmbedder(
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
