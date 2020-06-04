package lamp.nn

import lamp.autograd.{Variable, param}
import aten.ATen
import aten.TensorOptions

case class Meanshift(means: Variable, dim: List[Int]) extends Module {
  def parameters: Seq[(Variable, PTag)] = List(means -> Meanshift.Means)
  def forward(x: Variable): Variable = {
    val mean = x.mean(dim)
    (x - mean) + means
  }
}

object Meanshift {
  case object Means extends LeafTag
  def apply(
      size: List[Long],
      dim: List[Int] = List(0),
      tOpt: TensorOptions = TensorOptions.dtypeDouble
  ): Meanshift =
    Meanshift(
      dim = dim,
      means = param(ATen.zeros(size.toArray, tOpt))
    )
}
