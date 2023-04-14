package lamp.data

import cats.effect.IO
import lamp.Scope
import lamp.Device
import java.io.File
import java.io.FileOutputStream
import lamp.STen
import java.io.FileInputStream
import lamp.nn.LearningRateSchedule

/** Helpers to read and write training loop state
  */
object StateIO {

  private def readSimpleLoopStateDescriptor(
      s: schemas.SimpleLoopState,
      file: File,
      device: Device
  )(implicit scope: Scope) = {
    val model = Reader.readTensorData(s.model, file, device, false)
    val optim = Reader.readTensorData(s.optimizer, file, device, false)
    val minValid = s.minValidationLossModel.map { case (a, d) =>
      implicit val scope = Scope.free
      (a, Reader.readTensorData(d, file, device, false).map(_.value))
    }
    SimpleLoopState(
      model,
      optim,
      s.epoch,
      s.lastValidationLoss,
      s.minValidationLoss,
      minValid,
      s.learningCurve.map { case (epoch, train, smoothedValid, valid) =>
        (epoch, train, smoothedValid.flatMap { a => valid.map(b => (a, b)) })
      }
    )
  }
  private def readSWALoopStateDescriptor(
      s: schemas.SWALoopState,
      file: File,
      device: Device
  )(implicit scope: Scope) = {
    val model = Reader.readTensorData(s.model, file, device, false)
    val optim = Reader.readTensorData(s.optimizer, file, device, false)
    val avg = s.averagedModels.map { case d =>
      implicit val scope = Scope.free
      Reader.readTensorData(d, file, device, false).map(_.value)
    }
    SWALoopState(
      model,
      optim,
      s.epoch,
      s.lastValidationLoss,
      s.minValidationLoss,
      s.numberOfAveragedModels,
      avg,
      s.learningCurve
    )
  }

  /** Reads LoopState from file
    *
    * Returned value may be passed to training loop constructors to initialize
    * loop state
    *
    * @param file
    *   file on disk
    * @param device
    *   device onto which to load tensors
    * @return
    *   LoopState
    */
  def readFromFile(file: File, device: Device)(implicit
      scope: Scope
  ): LoopState = {

    val descriptor = {
      val is = new FileInputStream(file)
      try {
        com.github.plokhotnyuk.jsoniter_scala.core
          .readFromStream[schemas.LoopState](is)
      } finally {
        is.close
      }
    }
    descriptor match {
      case s: schemas.SimpleLoopState =>
        readSimpleLoopStateDescriptor(s, file, device)
      case s: schemas.SWALoopState =>
        readSWALoopStateDescriptor(s, file, device)
      case schemas.SimpleThenSWALoopState(simple, swa) =>
        SimpleThenSWALoopState(
          readSimpleLoopStateDescriptor(simple, file, device),
          swa.map(readSWALoopStateDescriptor(_, file, device))
        )

    }
  }

  private def simpleLoopStateDescriptor(
      s: SimpleLoopState,
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
    val minValidDescriptor = s.minValidationLossModel.map { case (a, ts) =>
      val location = s"${file.getName}.minvalidmodel"
      val channel = new FileOutputStream(
        new File(file.getParentFile(), location + ".tmp"),
        false
      ).getChannel
      val descriptor = Writer
        .writeTensorDataAndMakeDescriptor(
          tensors = ts.map(tensor => STen.owned(tensor)(Scope.free)),
          location,
          dataChannel = channel,
          bufferSize = bufferSize,
          initialByteOffset = 0
        )
        .toOption
        .get
      new File(file.getParentFile(), location + ".tmp").renameTo(
        new File(file.getParentFile(), location)
      )
      (a, descriptor)
    }
    new File(file.getParentFile(), optimizerLocation + ".tmp").renameTo(
      new File(file.getParentFile(), optimizerLocation)
    )
    new File(file.getParentFile(), modelLocation + ".tmp").renameTo(
      new File(file.getParentFile(), modelLocation)
    )

    schemas.SimpleLoopState(
      modelDescriptor,
      optimizerDescriptor,
      s.epoch,
      s.lastValidationLoss,
      s.minValidationLoss,
      minValidDescriptor,
      s.learningCurve.map { case (epoch, train, valid) =>
        (epoch, train, valid.map(_._1), valid.map(_._2))
      }
    )
  }
  private def swaLoopStateDescriptor(
      s: SWALoopState,
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
    new File(file.getParentFile(), optimizerLocation + ".tmp").renameTo(
      new File(file.getParentFile(), optimizerLocation)
    )
    new File(file.getParentFile(), modelLocation + ".tmp").renameTo(
      new File(file.getParentFile(), modelLocation)
    )
    val averageDescriptor = s.averagedModels.map { case ts =>
      val location = s"${file.getName}.averagemodel"
      val channel = new FileOutputStream(
        new File(file.getParentFile(), location),
        false
      ).getChannel
      val descriptor = Writer
        .writeTensorDataAndMakeDescriptor(
          tensors = ts.map(tensor => STen.owned(tensor)(Scope.free)),
          location,
          dataChannel = channel,
          bufferSize = bufferSize,
          initialByteOffset = 0
        )
        .toOption
        .get
      new File(file.getParentFile(), location + ".tmp").renameTo(
        new File(file.getParentFile(), location)
      )
      descriptor
    }

    schemas.SWALoopState(
      modelDescriptor,
      optimizerDescriptor,
      s.epoch,
      s.lastValidationLoss,
      s.minValidationLoss,
      s.numberOfAveragedModels,
      averageDescriptor,
      s.learningCurve
    )
  }

  /** Writes loop state into file
    */
  def writeToFile(
      file: File,
      state: LoopState,
      bufferSize: Int = 16384
  ): Unit = {

    val descriptor: schemas.LoopState = state match {
      case s: SimpleLoopState =>
        simpleLoopStateDescriptor(
          s,
          file,
          bufferSize
        )
      case s: SWALoopState =>
        swaLoopStateDescriptor(
          s,
          file,
          bufferSize
        )
      case s: SimpleThenSWALoopState =>
        schemas.SimpleThenSWALoopState(
          simple = simpleLoopStateDescriptor(
            s.simple,
            new File(file.getAbsolutePath() + ".simple"),
            bufferSize
          ),
          swa = s.swa.map(s =>
            swaLoopStateDescriptor(
              s,
              new File(file.getAbsolutePath() + ".swa"),
              bufferSize
            )
          )
        )
    }
    val tmp = new File(file.getAbsolutePath() + ".tmp")
    val fos = new java.io.FileOutputStream(tmp)
    try {
      com.github.plokhotnyuk.jsoniter_scala.core.writeToStream(descriptor, fos)
      tmp.renameTo(file)
    } finally { fos.close }

  }

  /** Returns a function which returns an IO writing the loop state to file
    */
  def stateToFile(file: File): LoopState => IO[Unit] = { (state: LoopState) =>
    IO.blocking {
      writeToFile(file, state, 16384)
    }
  }

  def reduceLROnPlateauStateToFile(
      file: File
  ): LearningRateSchedule.ReduceLROnPlateauState => IO[Unit] = {
    (state: LearningRateSchedule.ReduceLROnPlateauState) =>
      IO.blocking {
        val fos = new java.io.FileOutputStream(file)
        try {
          import lamp.data.schemas.LearningRateScheduleSchemas._
          com.github.plokhotnyuk.jsoniter_scala.core.writeToStream(state, fos)
        } finally { fos.close }
      }
  }
  def reduceLROnPlateauStateFromFile(
      file: File
  ): IO[LearningRateSchedule.ReduceLROnPlateauState] = IO.blocking {
    val fos = new java.io.FileInputStream(file)
    try {
      import lamp.data.schemas.LearningRateScheduleSchemas._
      com.github.plokhotnyuk.jsoniter_scala.core
        .readFromStream[LearningRateSchedule.ReduceLROnPlateauState](fos)
    } finally { fos.close }
  }
}
