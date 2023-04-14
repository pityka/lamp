package lamp.data.distributed

import lamp.Movable
import lamp.EmptyMovable
import lamp.STen
import java.io.File
import cats.effect.IO
import lamp.data.Writer
import java.io.FileOutputStream

case class LoopState(
    epoch: Int,
    lastValidationLoss: Option[Double],
    minValidationLoss: Option[Double],
    minValidationEpoch: Option[Int],
    learningCurve: List[(Int, Double, Option[(Double, Double)])]
)
object LoopState {
  implicit val movable: EmptyMovable[LoopState] = Movable.empty

  private def writeLoopStateDescriptor(
      s: LoopStateWithModelAndOptimizerData,
      file: File,
      bufferSize: Int
  ) = {
    val modelLocation = s"${file.getName}.model"
    val modelChannel = new FileOutputStream(
      new File(file.getParentFile(), modelLocation + ".tmp"),
      false
    ).getChannel
    val optimizerLocation = s"${file.getName}.optimizer"
    val optimizerChannel = new FileOutputStream(
      new File(file.getParentFile(), optimizerLocation + ".tmp"),
      false
    ).getChannel
    val modelDescriptor = Writer
      .writeTensorDataAndMakeDescriptor(
        tensors = s.model,
        modelLocation,
        dataChannel = modelChannel,
        bufferSize = bufferSize,
        initialByteOffset = 0
      )
      .toOption
      .get
    val optimizerDescriptor = Writer
      .writeTensorDataAndMakeDescriptor(
        tensors = s.optimizer,
        optimizerLocation,
        dataChannel = optimizerChannel,
        bufferSize = bufferSize,
        initialByteOffset = 0
      )
      .toOption
      .get
      val location = s"${file.getName}.minvalidmodel"
      val channel = new FileOutputStream(
        new File(file.getParentFile(), location + ".tmp"),
        false
      ).getChannel
      val minValidDescriptor = Writer
        .writeTensorDataAndMakeDescriptor(
          tensors = s.bestModel,
          location,
          dataChannel = channel,
          bufferSize = bufferSize,
          initialByteOffset = 0
        )
        .toOption
        .get
     
    new File(file.getParentFile(), optimizerLocation + ".tmp").renameTo(
      new File(file.getParentFile(), optimizerLocation)
    )
    new File(file.getParentFile(), modelLocation + ".tmp").renameTo(
      new File(file.getParentFile(), modelLocation)
    )

    lamp.data.schemas.SimpleLoopState(
      modelDescriptor,
      optimizerDescriptor,
      s.loopState.epoch,
      s.loopState.lastValidationLoss,
      s.loopState.minValidationLoss,
      Some(0 -> minValidDescriptor),
      s.loopState.learningCurve.map { case (epoch, train, valid) =>
        (epoch, train, valid.map(_._1), valid.map(_._2))
      }
    )
  }

  /** Writes loop state into file
    */
  def writeToFile(
      file: File,
      state: LoopStateWithModelAndOptimizerData,
      bufferSize: Int = 16384
  ): Unit = {

    val descriptor: lamp.data.schemas.LoopState = 
        writeLoopStateDescriptor(
          state,
          file,
          bufferSize
        )
   
    val tmp = new File(file.getAbsolutePath() + ".tmp")
    val fos = new java.io.FileOutputStream(tmp)
    try {
      com.github.plokhotnyuk.jsoniter_scala.core.writeToStream(descriptor, fos)
      tmp.renameTo(file)
    } finally { fos.close }

  }

  /** Returns a function which returns an IO writing the loop state to file
    */
  def stateToFile(file: File): LoopStateWithModelAndOptimizerData => IO[Unit] = { (state: LoopStateWithModelAndOptimizerData) =>
    IO.blocking {
      writeToFile(file, state, 16384)
    }
  }
}

case class LoopStateWithModelAndOptimizerData(
    loopState: LoopState,
    model: Seq[STen],
    optimizer: Seq[STen],
    bestModel: Seq[STen]
)
