package lamp.data

import scala.concurrent.duration._
import aten.TensorTrace
import aten.TensorTraceData

object TensorLogger {

  def formatLine(data: TensorTraceData) =
    s"${data.getShape.mkString(" ")}|${if (data.getCpu) "CPU" else "GPU"}|${if (data.getScalarType == 4) "L"
    else if (data.getScalarType == 6) "F"
    else "D"}"

  def formatStackTrace(data: TensorTraceData) = {
    data.getStackTrace.map(v => "\t\t" + v.toString).mkString("\n")
  }

  def queryActiveTensors() =
    aten.TensorTrace.list
      .map(v =>
        (
          v.getValue
        )
      )
      .toVector
  def makeStatistic() = {
    val currentTime = System.nanoTime
    val actives = queryActiveTensors()
    val lifetimes = actives.map {
      case data =>
        (data, (currentTime - data.getBirth) * 1e-6)
    }
    val histogram = {
      val tranches = List(
        0d -> 1e3,
        1e3 -> 60e3,
        60e3 -> 300e3,
        300e3 -> 3600e3,
        3600e3 -> Double.MaxValue
      )
      tranches.map {
        case (low, high) =>
          val count = lifetimes.count {
            case (_, duration) =>
              duration >= low && duration < high
          }
          (low, high, count)
      }
    }
    val leaks = lifetimes.filter {
      case (_, duration) =>
        duration > 300e3
    }
    (histogram, leaks)
  }
  def log(logger: String => Unit): Unit = {

    def format(d: Double) = d match {
      case 1d    => "1s"
      case 60d   => "1min"
      case 300d  => "5min"
      case 3600d => "1h"
      case _     => "Inf"
    }
    val (histogram, leaks) = makeStatistic()
    val string = s"Tensor lifetime histogram: ${histogram
      .map { case (_, high, count) => s"<${format(high / 1000)}:$count" }
      .mkString("|")}, total=${histogram.map(_._3).sum}. ${leaks
      .map {
        case (data, duration) =>
          s"${duration * 1e-6}s|${formatLine(data)}"
      }
      .mkString(",")}"

    logger(string)
  }

  def detailAllTensors(logger: String => Unit): Unit = {
    val currentTime = System.nanoTime
    val actives = queryActiveTensors()
    val lifetimes = actives.map {
      case data =>
        (data, (currentTime - data.getBirth) * 1e-6)
    }
    val strings = lifetimes.map {
      case (data, duration) =>
        s"\t${duration * 1e-6}s|${formatLine(data)}\n${formatStackTrace(data)}"
    }
    logger("\n" + strings.mkString("\n"))
  }

  def start(
      frequency: FiniteDuration = 5 seconds
  )(logger: String => Unit) = {
    TensorTrace.enable()
    @volatile
    var flag = true
    val t = new Thread {
      override def run = {
        while (flag) {
          log(logger)
          Thread.sleep(frequency.toMillis)
        }
      }
    }
    t.start()
    val stop = () => {
      log(logger)
      flag = false
    }
    TensorLogger(stop)

  }

}
case class TensorLogger(
    stop: () => Unit
) {
  def cancel() = {
    stop()
    TensorTrace.disable()
  }

}
