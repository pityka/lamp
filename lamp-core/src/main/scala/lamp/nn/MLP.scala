package lamp.nn

import lamp.autograd.{Variable, param}
import aten.{ATen, TensorOptions}
import lamp.autograd.TensorHelpers
import aten.Tensor

/** Factory for multilayer fully connected feed forward networks
  *
  * Returned network has the following repeated structure:
  * [linear -> batchnorm -> gelu -> dropout]*
  *
  * @param in input dimensions
  * @param out output dimensions
  * @param hidden list of hidden dimensions
  * @param dropout dropout applied to each block
  */
object MLP {
  def apply(
      in: Int,
      out: Int,
      hidden: Seq[Int],
      tOpt: TensorOptions,
      dropout: Double = 0d
  ): Sequential =
    Sequential(
      (List(in) ++ hidden ++ List(out)).sliding(2).toList.flatMap { group =>
        val in = group(0)
        val out = group(1)
        List(
          Linear(in = in, out = out, tOpt = tOpt, bias = true),
          BatchNorm(out, tOpt = tOpt),
          Fun(_.gelu),
          Dropout(dropout, training = true)
        )
      }: _*
    )

}
