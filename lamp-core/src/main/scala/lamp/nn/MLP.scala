package lamp.nn

import lamp.Sc
import lamp.STenOptions
import lamp.autograd.Variable
import lamp.nn.MLP.NormType.LayerNorm
import _root_.lamp.nn.MLP.NormType.BatchNorm

/** Factory for multilayer fully connected feed forward networks
  *
  * Returned network has the following repeated structure: [linear -> batchnorm
  * -> nonlinearity -> dropout]*
  *
  * The last block does not include the nonlinearity and the dropout.
  *
  * @param in
  *   input dimensions
  * @param out
  *   output dimensions
  * @param hidden
  *   list of hidden dimensions
  * @param dropout
  *   dropout applied to each block
  */
object MLP {
  sealed trait ActivationFunction
  case object Relu extends ActivationFunction
  case object Gelu extends ActivationFunction
  case object Swish1 extends ActivationFunction
  case object HardSwish extends ActivationFunction
  case object Sigmoid extends ActivationFunction

  sealed trait NormType
  object NormType {
    case object NoNorm extends NormType
    case object BatchNorm extends NormType
    case class LayerNorm(bias: Boolean, scale: Boolean) extends NormType
  }

  def apply[S: Sc](
      in: Int,
      out: Int,
      hidden: Seq[Int],
      tOpt: STenOptions,
      dropout: Double = 0d,
      lastNonLinearity: Boolean = false,
      activationFunction: ActivationFunction = Relu,
      norm: NormType = NormType.BatchNorm,
      numHeads: Int = 1,
      bias: Boolean = true
  ) = {

    def act() = activationFunction match {
      case Gelu      => Fun(scope => input => input.gelu(scope))
      case Relu      => Fun(scope => input => input.relu(scope))
      case Swish1    => Fun(scope => input => input.swish1(scope))
      case HardSwish => Fun(scope => input => input.hardSwish(scope))
      case Sigmoid   => Fun(scope => input => input.sigmoid(scope))
    }
    def makeNorm(normDim: Int): Sequential[Variable, EitherModule[
      Variable,
      Variable,
      lamp.nn.BatchNorm,
      lamp.nn.LayerNorm
    ]] = norm match {
      case NormType.NoNorm => Sequential()
      case NormType.BatchNorm =>
        Sequential(EitherModule(Left(lamp.nn.BatchNorm(normDim, tOpt = tOpt))))
      case NormType.LayerNorm(bias, scale) =>
        Sequential(
          EitherModule(
            Right(
              lamp.nn.LayerNorm(
                normalizedShape = List(normDim.toLong),
                tOpt = tOpt,
                bias = bias,
                scale = scale
                
              )
            )
          )
        )
    }

    val hasBias = norm match {
      case LayerNorm(true, _) | BatchNorm => false 
      case _ => bias 
    }

    sequence(
      Sequential(
        hidden.headOption.toList.map { out1 =>
          sequence(
            Linear(
              in = in,
              out = out1,
              tOpt = tOpt,
              bias = hasBias,
              numHeads = 1
            ),
            makeNorm(out1),
            act(),
            Dropout(dropout, training = true)
          )
        } ++
          hidden
            .sliding(2)
            .filter(_.size == 2)
            .toList
            .map { group =>
              val in = group(0)
              val out = group(1)
              sequence(
                Linear(
                  in = in,
                  out = out,
                  tOpt = tOpt,
                  bias = hasBias,
                  numHeads =
                    if (out % numHeads == 0 && in % numHeads == 0) numHeads
                    else 1
                ),
                makeNorm(out),
                act(),
                Dropout(dropout, training = true)
              )
            }: _*
      ),
      EitherModule(
        {
          val in1 = (List(in) ++ hidden).last
          if (lastNonLinearity)
            Left(
              sequence(
                Linear(
                  in = in1,
                  out = out,
                  tOpt = tOpt,
                  bias = hasBias,
                  numHeads =
                    if (out % numHeads == 0 && in1 % numHeads == 0) numHeads
                    else 1
                ),
                makeNorm(out),
                act(),
                Dropout(dropout, training = true)
              )
            )
          else
            Right(
              sequence(
                Linear(
                  in = in1,
                  out = out,
                  tOpt = tOpt,
                  bias = hasBias,
                  numHeads =
                    if (out % numHeads == 0 && in1 % numHeads == 0) numHeads
                    else 1
                ),
                makeNorm(out)
              )
            )
        }
      )
    )
  }

}
