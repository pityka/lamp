package lamp.tabular

import lamp.nn._
import aten.TensorOptions
import lamp.autograd.Variable
import aten.Tensor
import lamp.autograd.AllocatedVariablePool
import lamp.autograd.Concatenate

case class TabularEmbedding(
    categoricalEmbeddings: Seq[Embedding]
) extends GenericModule[(Seq[Variable], Variable), Variable] {
  override def state =
    categoricalEmbeddings.flatMap(_.state)
  def forward(x: (Seq[Variable], Variable)) = {
    val (categoricals, numericals) = x
    assert(
      categoricals.size == categoricalEmbeddings.size,
      s"${categoricals.size} vs ${categoricalEmbeddings.size}"
    )
    val embeddedCategoricals = categoricals.zip(categoricalEmbeddings).map {
      case (v, embedding) =>
        embedding.forward(v).view(List(v.shape.head.toInt, -1))
    }
    Concatenate(embeddedCategoricals :+ numericals, dim = 1).value
  }

}

object TabularEmbedding {
  def make(
      categoricalClassesWithEmbeddingDimensions: Seq[(Int, Int)],
      tOpt: TensorOptions
  )(implicit pool: AllocatedVariablePool) =
    TabularEmbedding(
      categoricalEmbeddings = categoricalClassesWithEmbeddingDimensions.map {
        case (classes, size) =>
          Embedding(classes, size, tOpt)
      }
    )

  implicit def trainingMode =
    TrainingMode.make[TabularEmbedding](
      m =>
        TabularEmbedding(
          m.categoricalEmbeddings.map(_.asEval)
        ),
      m =>
        TabularEmbedding(
          m.categoricalEmbeddings.map(_.asTraining)
        )
    )
  implicit def load =
    Load.make[TabularEmbedding](m =>
      t => {
        def loop(
            ts: Seq[Tensor],
            mods: Seq[Embedding],
            acc: List[(Embedding, Seq[Tensor])]
        ): (Seq[Tensor], Seq[(Embedding, Seq[Tensor])]) =
          if (mods.size == 0) (ts, acc)
          else {
            loop(
              ts.drop(mods.head.state.size),
              mods.tail,
              (mods.head, ts.take(mods.head.state.size)) :: acc
            )
          }

        val (_, zippedEmbeddingTensors) =
          loop(t, m.categoricalEmbeddings, Nil)

        val loadedCategoricals =
          zippedEmbeddingTensors.reverse.filter(_._2.nonEmpty).map {
            case (emb, ts) => emb.load(ts)
          }
        TabularEmbedding(
          loadedCategoricals
        )
      }
    )

}
