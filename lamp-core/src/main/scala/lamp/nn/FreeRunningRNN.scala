package lamp.nn
import lamp.autograd.Variable
import lamp.Sc

/**
  * Wraps a (sequence x batch) long -> (sequence x batch x dim) double stateful module
  * and runs in it greedy (argmax) generation mode over `timeSteps` steps.
  *
  */
case class FreeRunningRNN[T, M <: StatefulModule[Variable, Variable, T]](
    module: M with StatefulModule[Variable, Variable, T],
    timeSteps: Int
) extends StatefulModule[Variable, Variable, T] {

  def state = module.state

  def forward[S: Sc](x: (Variable, T)) = {
    val batchSize = x._1.shape(1)
    val (outputs, lastState) = loop(x._1, x._2, timeSteps, Nil)
    (
      Variable
        .concatenateAddNewDim(outputs)
        .view(List(timeSteps, batchSize.toInt, -1)),
      lastState
    )
  }
  def loop[S: Sc](
      lastOutput: Variable,
      lastState: T,
      n: Int,
      buffer: Seq[Variable]
  ): (Seq[Variable], T) = {
    if (n == 0) (buffer, lastState)
    else {
      val (output, state) =
        module.forward((lastOutput, lastState))

      val lastChar = if (output.shape(0) > 1) {
        val lastTimeStep1 =
          output.select(0, output.shape(0) - 1)

        lastTimeStep1.view((1L :: lastTimeStep1.shape).map(_.toInt))

      } else output

      val nextInput =
        lastChar.argmax(2, false).detached

      loop(
        nextInput,
        state,
        n - 1,
        buffer :+ lastChar
      )
    }
  }
}
object FreeRunningRNN {
  implicit def trainingMode[T, M <: StatefulModule[Variable, Variable, T]: TrainingMode] =
    TrainingMode.make[FreeRunningRNN[T, M]](
      m => m.copy(module = m.module.asEval),
      m => m.copy(module = m.module.asTraining)
    )
  implicit def is[T, M <: StatefulModule[Variable, Variable, T]](
      implicit st: InitState[M, T]
  ) =
    InitState.make[FreeRunningRNN[T, M], T](m => m.module.initState)
  implicit def load[T, M <: StatefulModule[Variable, Variable, T]: Load] =
    Load.make[FreeRunningRNN[T, M]] { m => tensors => m.module.load(tensors) }
}
