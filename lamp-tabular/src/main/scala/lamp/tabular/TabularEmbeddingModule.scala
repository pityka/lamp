package lamp.tabular

import lamp.nn._
import lamp.autograd.Variable
import lamp.Sc
import lamp.Scope
import lamp.STen
import lamp.STenOptions

case class TabularEmbedding(
    categoricalEmbeddings: Seq[Embedding]
) extends GenericModule[(Seq[Variable], Variable), Variable] {
  override def state =
    categoricalEmbeddings.flatMap(_.state)
  def forward[S: Sc](x: (Seq[Variable], Variable)) = {
    val (categoricals, numericals) = x
    assert(
      categoricals.size == categoricalEmbeddings.size,
      s"${categoricals.size} vs ${categoricalEmbeddings.size}"
    )
    val embeddedCategoricals = categoricals.zip(categoricalEmbeddings).map {
      case (v, embedding) =>
        embedding.forward(v).view(List(v.shape.head.toInt, -1))
    }
    Variable.cat(embeddedCategoricals :+ numericals, dim = 1)
  }

}

object TabularEmbedding {
  def make(
      categoricalClassesWithEmbeddingDimensions: Seq[(Int, Int)],
      tOpt: STenOptions
  )(implicit scope: Scope) =
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
            ts: Seq[STen],
            mods: Seq[Embedding],
            acc: List[(Embedding, Seq[STen])]
        ): (Seq[STen], Seq[(Embedding, Seq[STen])]) =
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

        zippedEmbeddingTensors.reverse.filter(_._2.nonEmpty).foreach {
          case (emb, ts) => emb.load(ts)
        }

      }
    )

}
