package lamp.nn

import lamp.autograd.AllocatedVariablePool
import aten.TensorOptions

/** Factory for multilayer fully connected feed forward networks
  *
  * Returned network has the following repeated structure:
  * [linear -> batchnorm -> gelu -> dropout]*
  *
  * The last block does not include the nonlinearity and the dropout.
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
      dropout: Double = 0d,
      lastNonLinearity: Boolean = false
  )(implicit pool: AllocatedVariablePool) =
    sequence(
      Sequential(
        (List(in) ++ hidden)
          .sliding(2)
          .filter(_.size == 2)
          .toList
          .map { group =>
            val in = group(0)
            val out = group(1)
            sequence(
              Linear(in = in, out = out, tOpt = tOpt, bias = false),
              BatchNorm(out, tOpt = tOpt),
              Fun(_.gelu),
              Dropout(dropout, training = true)
            )
          }: _*
      ),
      EitherModule(
        if (lastNonLinearity)
          Left(
            sequence(
              Linear(
                in = (List(in) ++ hidden).last,
                out = out,
                tOpt = tOpt,
                bias = false
              ),
              BatchNorm(out, tOpt = tOpt),
              Fun(_.gelu),
              Dropout(dropout, training = true)
            )
          )
        else
          Right(
            sequence(
              Linear(
                in = (List(in) ++ hidden).last,
                out = out,
                tOpt = tOpt,
                bias = false
              ),
              BatchNorm(out, tOpt = tOpt)
            )
          )
      )
    )

}
