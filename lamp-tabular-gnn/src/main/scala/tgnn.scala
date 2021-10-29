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
case class SingleVariableColumn(constant: Constant) extends ColumnDataType

object ColumnDataType {
  implicit val movable: Movable[ColumnDataType] =
    Movable.by[ColumnDataType, Constant](_ match {
      case SingleVariableColumn(constant) => constant
    })
}

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
    columns: Seq[ColumnDataType],
    length: Long
) {
  def takeRows(v: Vec[Long], device: Device)(implicit
      scope: Scope
  ): TableInColumns = Scope { implicit scope =>
    val currentDevice = columns.head match {
      case SingleVariableColumn(variable) =>
        variable.value.device
    }
    val indexS = STen.fromLongVec(v, currentDevice)
    TableInColumns(
      columns = columns.map {
        _ match {
          case SingleVariableColumn(variable) =>
            SingleVariableColumn(
              const(
                variable.value
                  .indexSelect(dim = 0, index = indexS)
                  .copyToDevice(device)
              )
            )

        }
      },
      length = v.length
    )

  }
}
object TableInColumns {
  implicit val movable: Movable[TableInColumns] =
    Movable.by[TableInColumns, Seq[ColumnDataType]](_.columns)
}

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
case class Batch(
    tables: Seq[TableInColumns],
    edgeI: STen,
    edgeJ: STen,
    vertexPooling: STen,
    target: STen
) {
  def subset[S: Sc](select: Vec[Long], device: Device): Batch = {
    val sorted = select.sorted

    val tableLengths = tables.map(_.length)
    val tableOffsets = tableLengths.scanLeft(0L)(_ + _)

    val selectedTables = tableOffsets.zip(tableLengths).zipWithIndex.map {
      case ((tableOffset, tableLength), tableIdx) =>
        tables(tableIdx).takeRows(
          sorted
            .filter(x => x >= tableOffset && x < tableOffset + tableLength)
            .map(_ - tableOffset),
          device
        )
    }

    val sortedI = Index(sorted)

    val ni1 = edgeI.toLongVec.map(x => sortedI.getFirst(x).toLong)
    val nj1 = edgeJ.toLongVec.map(x => sortedI.getFirst(x).toLong)
    val keep = Index(ni1.find(_ >= 0))
      .intersect(Index(nj1.find(_ >= 0)))
      .index
      .toVec
      .toArray
    val ni = ni1.take(keep)
    val nj = nj1.take(keep)
    val vp1 = vertexPooling.toLongVec
      .map(x => sortedI.getFirst(x).toLong)

    val keepVp = vp1.find(_ >= 0)

    val vp = vp1.take(keep.toArray)

    Batch(
      selectedTables,
      STen.fromLongVec(ni, device),
      STen.fromLongVec(nj, device),
      STen.fromLongVec(vp, device),
      target.indexSelect(
        dim = 0,
        index = STen.fromLongVec(keepVp.map(_.toLong), device)
      )
    )

  }
}
object Batch {

  def bfs(
      start: Vec[Long],
      indexI: Index[Long],
      indexJ: Index[Long],
      depth: Int
  ): Vec[Long] = {

    val edgeI = indexI.toVec
    val edgeJ = indexJ.toVec

    def getChildren(nodes: Vec[Long]): Vec[Long] =
      (edgeI.take(indexI(nodes.toArray)) concat
        edgeJ.take(indexJ(nodes.toArray)))

    def substract(v1: Index[Long], v2: Vec[Long]): Vec[Long] =
      v2.filter(l => !v1.contains(l))

    def loop(
        unexplored: Vec[Long],
        explored: Index[Long],
        currentDepth: Int
    ): Vec[Long] = {
      if (unexplored.isEmpty || currentDepth == depth) explored.toVec
      else {
        val children = getChildren(unexplored)
        val novel = substract(explored, children)
        loop(novel, explored.concat(Index(novel)), currentDepth + 1)
      }
    }

    loop(start, Index.empty, 0)

  }

  def subsetBySubject(
      minibatchSize: Int,
      subjects: Vec[Int],
      tables: Tables,
      depth: Int,
      rng: Option[org.saddle.spire.random.Generator]
  )(implicit scope: Scope): BatchStream[Batch, Int] = {
    val batch = Batch.makeBatch(tables, STenOptions.f)
    val indexI = Index(batch.edgeI.toLongVec)
    val indexJ = Index(batch.edgeJ.toLongVec)
    def makeNonEmptyBatch(idx: Array[Int], device: Device) = {
      Scope.inResource.map { implicit scope =>
        val selected = bfs(
          start = idx.toVec.map(_.toLong),
          indexI = indexI,
          indexJ = indexJ,
          depth = depth
        )
        val s = batch.subset(selected, device)

        StreamControl(
          (s, s.target)
        )
      }
    }

    val idx =
      rng
        .map(rng =>
          array
            .shuffle(subjects.toArray, rng)
            .grouped(minibatchSize)
            .toList
        )
        .getOrElse(
          subjects.toArray
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
    val targets = STen.cat(tables.targets, dim = 0)
    Batch(dataTables, i, j, vertexPooling, targets)
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

  def inferEmbeddingTypes(
      dataTables: Seq[TableInColumns]
  ): Seq[Seq[EmbeddingType]] = {
    dataTables.map { columns =>
      columns.columns.map { column =>
        EmbeddingType.inferFromColumn(column)
      }
    }
  }

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

object EmbeddingType {
  case class Embedder(numClasses: Int, outDim: Int) extends EmbeddingType
  case class Linear(inDim: Int, outDim: Int) extends EmbeddingType
// case class NLP(length: Int, outDim: Int) extends EmbeddingType
  def inferFromColumn(column: ColumnDataType): EmbeddingType = column match {
    case SingleVariableColumn(constant) =>
      constant.shape.size match {
        case 1 =>
          Scope.leak { implicit scope =>
            val randomIndex = STen.randint(
              constant.shape(0),
              List(math.min(constant.shape(0), 10000)),
              STenOptions.l
            )
            val sample = constant.value.indexSelect(dim = 0, randomIndex)

            // test this
            val uniqueCount = sample
              .unique(
                dim = 0,
                sorted = false,
                returnInverse = false,
                returnCounts = false
              )
              ._1
              .shape(0)
              .toInt
            val isFloat = constant.options.isFloat || constant.options.isDouble
            if (uniqueCount > 2000 || isFloat) Linear(inDim = 1, outDim = 8)
            else
              Embedder(
                numClasses = uniqueCount,
                outDim = math.max(1, math.sqrt(uniqueCount) / 8).toInt * 8
              )

          }

        case 2 =>
          Linear(
            inDim = constant.shape(1).toInt,
            outDim = math.max(1, constant.shape(1).toInt / 8) * 16
          )
        case _ => ???
      }

  }
}

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
        case EmbeddingType.Embedder(_, outDim) => outDim
        case EmbeddingType.Linear(_, outDim)   => outDim
      }
    }.sum
    TableEmbedder(
      columnEmbeddingTypes map { attr =>
        attr match {
          case EmbeddingType.Embedder(numClasses, outDim) =>
            ColumnEmbeddingModule(
              lamp.nn
                .Embedding(
                  classes = numClasses,
                  dimensions = outDim,
                  tOpt = tOpt
                )
            )
          case EmbeddingType.Linear(in, out) =>
            ColumnLinearModule(
              lamp.nn.Linear(in = in, out = out, tOpt = tOpt, bias = true)
            )
        }
      },
      lamp.nn.Linear(in = d, out = outDim, tOpt, bias = true)
    )
  }
}
