package lamp.data

import cats.effect.IO
import java.io.File
import java.io.FileOutputStream
import lamp.STen.OwnedSyntax
import lamp.Scope
import lamp.Device

object StateIO {

  def readFromFile(file: String, device: Device)(implicit
      scope: Scope
  ): State = {
    val descriptor =
      ujson.read({
        val src = scala.io.Source.fromFile(file + ".json")
        val str = src.mkString
        src.close
        str
      })
    val tpe = descriptor("tpe").str

    tpe match {
      case "simplethenswa" =>
        val simple =
          if (new File(file + ".simple.json").canRead)
            Some(
              readFromFile(file + ".simple", device)
                .asInstanceOf[SimpleLoopState]
            )
          else None
        val swa =
          if (new File(file + ".swa.json").canRead)
            Some(readFromFile(file + ".swa", device).asInstanceOf[SWALoopState])
          else None
        SimpleThenSWALoopState(simple, swa)
      case "swa" =>
        val epoch = descriptor("epoch").num.toInt
        val lastValidationLoss = {
          descriptor("lastValidationLoss") match {
            case ujson.Num(x) => Some(x)
            case ujson.Null   => None
            case _            => ???
          }
        }
        val learningCurve = descriptor("learningCurve").arr.map { elems =>
          (
            elems.arr(0).num.toInt,
            elems.arr(1).num.toDouble,
            elems.arr(2) match {
              case ujson.Null   => None
              case ujson.Num(x) => Some(x)
              case _            => ???
            }
          )
        }.toList
        val minValidationLoss = {
          descriptor("minValidationLoss") match {
            case ujson.Num(x) => Some(x)
            case ujson.Null   => None
            case _            => ???
          }
        }
        val numberOfAveragedModels =
          descriptor("numberOfAveragedModels").num.toInt

        val model =
          Reader
            .readTensorsFromFile(new File(file + ".model"), device)
            .toOption
            .get
            .map(_.owned)
        val optimizer =
          Reader
            .readTensorsFromFile(new File(file + ".optimizer"), device)
            .toOption
            .get
            .map(_.owned)
        val averagedModels =
          Reader
            .readTensorsFromFile(
              new File(file + ".averagedModels"),
              device
            )
            .toOption
            .get
        SWALoopState(
          model = model,
          optimizer = optimizer,
          epoch = epoch,
          lastValidationLoss = lastValidationLoss,
          minValidationLoss = minValidationLoss,
          numberOfAveragedModels = numberOfAveragedModels,
          averagedModels =
            if (averagedModels.isEmpty) None else Some(averagedModels),
          learningCurve = learningCurve
        )
      case "simple" =>
        val epoch = descriptor("epoch").num.toInt
        val lastValidationLoss = descriptor("lastValidationLoss") match {
          case ujson.Num(x) => Some(x)
          case ujson.Null   => None
          case _            => ???
        }
        val learningCurve = descriptor("learningCurve").arr.map { elems =>
          (
            elems.arr(0).num.toInt,
            elems.arr(1).num.toDouble,
            elems.arr(2) match {
              case ujson.Null   => None
              case ujson.Num(x) => Some(x)
              case _            => ???
            }
          )
        }.toList
        val minValidationLoss = descriptor("minValidationLoss") match {
          case ujson.Num(x) => Some(x)
          case ujson.Null   => None
          case _            => ???
        }
        val model =
          Reader
            .readTensorsFromFile(new File(file + ".model"), device)
            .toOption
            .get
            .map(_.owned)
        val optimizer =
          Reader
            .readTensorsFromFile(new File(file + ".optimizer"), device)
            .toOption
            .get
            .map(_.owned)
        val minValidationLossModel =
          Reader
            .readTensorsFromFile(
              new File(file + ".minValidationLossModel"),
              device
            )
            .toOption
            .get
        SimpleLoopState(
          model = model,
          optimizer = optimizer,
          epoch = epoch,
          lastValidationLoss = lastValidationLoss,
          minValidationLoss = minValidationLoss,
          minValidationLossModel =
            if (minValidationLossModel.isEmpty) None
            else Some((0, minValidationLossModel)),
          learningCurve = learningCurve
        )
    }

  }

  def writeToFile(file: String, state: State): Unit = {
    state match {
      case state: SimpleThenSWALoopState =>
        val descriptor = ujson
          .write(
            ujson.Obj(
              "tpe" -> ujson.Str("simplethenswa")
            )
          )

        val fis =
          new FileOutputStream(new File(file + ".json"))
        fis.write(descriptor.getBytes("UTF-8"))
        fis.close
        state.simple.foreach(state => writeToFile(file + ".simple", state))
        state.swa.foreach(state => writeToFile(file + ".swa", state))
      case state: SimpleLoopState =>
        val descriptor = ujson
          .write(
            ujson.Obj(
              "tpe" -> ujson.Str("simple"),
              "epoch" -> ujson.Num(state.epoch),
              "lastValidationLoss" ->
                state.lastValidationLoss.map(ujson.Num).getOrElse(ujson.Null),
              "minValidationLoss" ->
                state.minValidationLoss.map(ujson.Num).getOrElse(ujson.Null),
              "learningCurve" ->
                state.learningCurve.map { case (a, b, c) =>
                  List(
                    ujson.Num(a),
                    ujson.Num(b),
                    c.map(ujson.Num).getOrElse(ujson.Null)
                  )

                }
            )
          )

        val fis =
          new FileOutputStream(new File(file + ".json"))
        fis.write(descriptor.getBytes("UTF-8"))
        fis.close

        {
          val channel = new FileOutputStream(
            new File(file + ".model")
          ).getChannel()
          Writer.writeTensorsIntoChannel(state.model.map(_.value), channel)
          channel.close()
        }

        {
          val channel = new FileOutputStream(
            new File(file + ".optimizer")
          ).getChannel()
          Writer.writeTensorsIntoChannel(
            state.optimizer.map(_.value),
            channel
          )
          channel.close()
        }

        {
          val channel = new FileOutputStream(
            new File(file + ".minValidationLossModel")
          ).getChannel()
          Writer.writeTensorsIntoChannel(
            state.minValidationLossModel
              .map(_._2)
              .toList
              .flatten,
            channel
          )
          channel.close()
        }
      case state: SWALoopState =>
        val descriptor = ujson
          .write(
            ujson.Obj(
              "tpe" -> ujson.Str("swa"),
              "epoch" -> ujson.Num(state.epoch),
              "lastValidationLoss" ->
                state.lastValidationLoss.map(ujson.Num).getOrElse(ujson.Null),
              "minValidationLoss" ->
                state.minValidationLoss.map(ujson.Num).getOrElse(ujson.Null),
              "numberOfAveragedModels" -> ujson.Num(
                state.numberOfAveragedModels
              ),
              "learningCurve" ->
                state.learningCurve.map { case (a, b, c) =>
                  List(
                    ujson.Num(a),
                    ujson.Num(b),
                    ujson.Num(c.getOrElse(Double.NaN))
                  )

                }
            )
          )

        val fis =
          new FileOutputStream(new File(file + ".json"))
        fis.write(descriptor.getBytes("UTF-8"))
        fis.close()

        {
          val channel = new FileOutputStream(
            new File(file + ".model")
          ).getChannel()
          Writer.writeTensorsIntoChannel(state.model.map(_.value), channel)
          channel.close()
        }

        {
          val channel = new FileOutputStream(
            new File(file + ".optimizer")
          ).getChannel()
          Writer.writeTensorsIntoChannel(
            state.optimizer.map(_.value),
            channel
          )
          channel.close()
        }

        {
          val channel = new FileOutputStream(
            new File(file + ".averagedModels")
          ).getChannel()
          Writer.writeTensorsIntoChannel(
            state.averagedModels.toList.flatten,
            channel
          )
          channel.close()
        }

    }
  }

  def stateToFile(file: String) = { (state: State) =>
    IO { writeToFile(file, state) }
  }
}
