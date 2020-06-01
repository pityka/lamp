package lamp.data

import cats.effect.IO
import aten.Tensor

trait UnbatchedDataset {
  def nextExample: IO[(Tensor, Tensor)]
}
trait UnbatchedIndexedDataset {
  def get(i: Long): IO[(Tensor, Tensor)]
}

trait BatchStream {
  def nextBatch: IO[(Tensor, Tensor)]
}
