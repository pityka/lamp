package lamp.extratrees

sealed trait ClassificationTree
case class ClassificationLeaf(targetDistribution: Seq[Double])
    extends ClassificationTree
object ClassificationLeaf {
  import upickle.default.{ReadWriter => RW, macroRW}
  implicit val rw: RW[ClassificationLeaf] = macroRW
}
case class ClassificationNonLeaf(
    left: ClassificationTree,
    right: ClassificationTree,
    splitFeature: Int,
    cutpoint: Double
) extends ClassificationTree
object ClassificationNonLeaf {
  import upickle.default.{ReadWriter => RW, macroRW}
  implicit val rw: RW[ClassificationNonLeaf] = macroRW
}

object ClassificationTree {
  import upickle.default.{ReadWriter => RW, macroRW}
  implicit val rw: RW[ClassificationTree] = macroRW
}

sealed trait RegressionTree
case class RegressionLeaf(targetMean: Double) extends RegressionTree
case class RegressionNonLeaf(
    left: RegressionTree,
    right: RegressionTree,
    splitFeature: Int,
    cutpoint: Double
) extends RegressionTree
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
