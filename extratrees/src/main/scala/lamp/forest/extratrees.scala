package lamp.extratrees

sealed trait ClassificationTree
case class ClassificationLeaf(targetDistribution: Seq[Double])
    extends ClassificationTree {

  override def toString =
    s"ClassificationTree(targetDistribution=$targetDistribution)"
}
object ClassificationLeaf {
  import upickle.default.{ReadWriter => RW, macroRW}
  implicit val rw: RW[ClassificationLeaf] = macroRW
}
case class ClassificationNonLeaf(
    left: ClassificationTree,
    right: ClassificationTree,
    splitFeature: Int,
    cutpoint: Double,
    splitMissingIsLess: Boolean
) extends ClassificationTree {
  override def toString =
    s"ClassificationTree(left=$left,right=$right,splitFeatures=$splitFeature,cutpoint=$cutpoint,splitMissingIsLess=$splitMissingIsLess)"
}
object ClassificationNonLeaf {
  import upickle.default.{ReadWriter => RW, macroRW}
  implicit val rw: RW[ClassificationNonLeaf] = macroRW
}

object ClassificationTree {
  import upickle.default.{ReadWriter => RW, macroRW}
  implicit val rw: RW[ClassificationTree] = macroRW
}

sealed trait RegressionTree
case class RegressionLeaf(targetMean: Double) extends RegressionTree {

  override def toString = s"RegressionLeaf(targetMean=$targetMean)"
}

case class RegressionNonLeaf(
    left: RegressionTree,
    right: RegressionTree,
    splitFeature: Int,
    cutpoint: Double,
    splitMissingIsLess: Boolean
) extends RegressionTree {
  override def toString =
    s"RegressionNonLeaf(left=$left,right=$right,splitFeatures=$splitFeature,cutpoint=$cutpoint,splitMissingIsLess=$splitMissingIsLess)"
}
object RegressionLeaf {
  import upickle.default.{ReadWriter => RW, macroRW}
  implicit val rw: RW[RegressionLeaf] = macroRW
}

object RegressionNonLeaf {
  import upickle.default.{ReadWriter => RW, macroRW}
  implicit val rw: RW[RegressionNonLeaf] = macroRW
}

object RegressionTree {
  import upickle.default.{ReadWriter => RW, macroRW}
  implicit val rw: RW[RegressionTree] = macroRW
}
