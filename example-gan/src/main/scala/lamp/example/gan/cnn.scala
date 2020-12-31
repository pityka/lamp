package lamp.example.gan

import lamp.nn._
import lamp.Scope
import lamp.STenOptions

object Cnn {

  def block(in: Int, out: Int, tOpt: STenOptions)(implicit scope: Scope) =
    Sequential(
      Conv2D(
        inChannels = in,
        outChannels = out,
        kernelSize = 4,
        padding = 1,
        stride = 2,
        tOpt = tOpt
      ),
      BatchNorm2D(out, tOpt),
      Fun(implicit scope => _.leakyRelu(0.2))
    )
  def generatorblock(in: Int, out: Int, tOpt: STenOptions)(
      implicit scope: Scope
  ) =
    Sequential(
      Conv2DTransposed(
        inChannels = in,
        outChannels = out,
        kernelSize = 4,
        padding = 1,
        stride = 2,
        tOpt = tOpt
      ),
      BatchNorm2D(out, tOpt),
      Fun(implicit scope => _.relu)
    )

  def discriminator(
      in: Int,
      tOpt: STenOptions
  )(implicit pool: Scope) =
    Sequential(
      Conv2D(
        inChannels = in,
        outChannels = 32,
        kernelSize = 4,
        tOpt = tOpt,
        padding = 1,
        stride = 2
      ),
      block(32, 64, tOpt),
      block(64, 128, tOpt),
      block(128, 256, tOpt),
      Conv2D(
        inChannels = 256,
        outChannels = 1,
        kernelSize = 4,
        tOpt = tOpt,
        padding = 1,
        stride = 2
      ),
      Fun(implicit pool => _.flatten)
    )
  def generator(
      in: Int,
      tOpt: STenOptions
  )(implicit scope: Scope) =
    Sequential(
      generatorblock(in, 256, tOpt),
      generatorblock(256, 128, tOpt),
      generatorblock(128, 64, tOpt),
      generatorblock(64, 32, tOpt),
      Conv2DTransposed(
        inChannels = 32,
        outChannels = 3,
        kernelSize = 4,
        tOpt = tOpt,
        padding = 1,
        stride = 2
      ),
      Fun(implicit scope => v => v.sigmoid * 255)
    )
}
