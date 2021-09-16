package lamp.data

import scala.concurrent.duration._
import aten.TensorTrace
import aten.TensorTraceData
import aten.TensorOptionsTrace

/** Utility to periodically log active tensors See
  * [[lamp.data.TensorLogger#start]]
  */
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
  def queryActiveTensorOptions() =
    aten.TensorOptionsTrace.list
      .map(v =>
        (
          v.getValue
        )
      )
      .toVector
  def makeStatistic(nanoTime: Long, actives: Seq[TensorTraceData])(
      filter: (TensorTraceData, Double) => Boolean
  ) = {
    val currentTime = nanoTime

    val totalActiveBytePerDevice = actives
      .map { t =>
        val numel = t.getShape().toList.foldLeft(1L)(_ * _)
        val elementSize = t.getScalarType() match {
          case 4 => 8
          case 5 => 2
          case 6 => 4
          case 7 => 8
          case _ => 1
        }
        val numbytes = numel * elementSize
        (t.getCpu(), numbytes)
      }
      .groupMapReduce(_._1)(_._2)(_ + _)

    val lifetimes = actives
      .map { case data =>
        (data, (currentTime - data.getBirth) * 1e-6)
      }
      .filter(filter.tupled)
    val histogram = {
      val tranches = List(
        0d -> 1e3,
        1e3 -> 60e3,
        60e3 -> 300e3,
        300e3 -> 3600e3,
        3600e3 -> Double.MaxValue
      )
      tranches.map { case (low, high) =>
        val count = lifetimes.count { case (_, duration) =>
          duration >= low && duration < high
        }
        (low, high, count)
      }
    }

    (histogram, lifetimes, totalActiveBytePerDevice)
  }
  def logTensors(
      logger: String => Unit,
      filter: (TensorTraceData, Double) => Boolean,
      detailMinMs: Double,
      detailMaxMs: Double,
      detailNum: Int
  ): Unit = {
    val now = System.nanoTime()
    val data = queryActiveTensors()
    val str = makeLog(now, data, filter, detailMinMs, detailMaxMs, detailNum)
    logger("Tensors - " + str)
  }
  def logTensorOptions(
      logger: String => Unit,
      filter: (TensorTraceData, Double) => Boolean,
      detailMinMs: Double,
      detailMaxMs: Double,
      detailNum: Int
  ): Unit = {
    val now = System.nanoTime()
    val data = queryActiveTensorOptions()
    val str = makeLog(now, data, filter, detailMinMs, detailMaxMs, detailNum)
    logger("TensorOptions - " + str)
  }
  def makeLog(
      nanoTime: Long,
      data: Seq[TensorTraceData],
      filter: (TensorTraceData, Double) => Boolean,
      detailMinMs: Double,
      detailMaxMs: Double,
      detailNum: Int
  ): String = {

    def format(d: Double) = d match {
      case 1d    => "1s"
      case 60d   => "1min"
      case 300d  => "5min"
      case 3600d => "1h"
      case _     => "Inf"
    }
    val (histogram, actives, totalBytes) = makeStatistic(nanoTime, data)(filter)
    val string = s" lifetime histogram: ${histogram
      .map { case (_, high, count) => s"<${format(high / 1000)}:$count" }
      .mkString("|")}, total=${histogram.map(_._3).sum}. \nTotal bytes on CPU: ${totalBytes
      .get(true)
      .getOrElse(0L)}. Total bytes on all GPUs: ${totalBytes.get(false).getOrElse(0L)}.  ${actives
      .filter { case (_, ms) => ms >= detailMinMs && ms <= detailMaxMs }
      .take(detailNum)
      .map { case (data, duration) =>
        s"${duration * 1e-3}s|${formatLine(data)}\n${formatStackTrace(data)}"
      }
      .mkString("\n", ";\n", "")}"

    string
  }

  def detailAllTensors(logger: String => Unit): Unit = {
    val currentTime = System.nanoTime
    val actives = queryActiveTensors()
    val lifetimes = actives.map { case data =>
      (data, (currentTime - data.getBirth) * 1e-6)
    }
    val strings = lifetimes.map { case (data, duration) =>
      s"\t${duration * 1e-3}s|${formatLine(data)}\n${formatStackTrace(data)}"
    }
    logger("\n" + strings.mkString("\n"))
  }
  def detailAllTensorOptions(logger: String => Unit): Unit = {
    val currentTime = System.nanoTime
    val actives = queryActiveTensorOptions()
    val lifetimes = actives.map { case data =>
      (data, (currentTime - data.getBirth) * 1e-6)
    }
    val strings = lifetimes.map { case (data, duration) =>
      s"\t${duration * 1e-3}s|${formatLine(data)}\n${formatStackTrace(data)}"
    }
    logger("\n" + strings.mkString("\n"))
  }

  /** Log active tensors with a given frequency
    *
    * The log string itself consists of a summary statistics of active tensors
    * and a list of detailed information on some number of active tensors. The
    * detailed information shows the stack trace where the tensor was allocated.
    * The tensors eligible for detail are specified by a filter on their
    * lifetime.
    *
    * This method will spawn a new Thread, which repeatedly logs and sleeps
    * until not canceled.
    *
    * @param frequency
    *   the frequency with which the logging will occur
    * @param logger
    *   a side effecting lambda which executes the log
    * @param filter
    *   a predicate taking some information about the tensor and the lifetime of
    *   the tensor in milliseconds
    * @param detailMinMs
    *   minimum lifetime in milliseconds for a tensor to be eligible for
    *   detailed logging
    * @param detailMaxMs
    *   maximum lifetime in milliseconds for a tensor to be eligible for
    *   detailed logging
    * @param detailNum
    *   max number of tensors in the detailed section
    * @return
    *   an instance of TensorLogger whose sole purpose is its `cancel()` method
    *   which stops the logger
    */
  def start(
      frequency: FiniteDuration = 5 seconds
  )(
      logger: String => Unit,
      filter: (TensorTraceData, Double) => Boolean,
      detailMinMs: Double,
      detailMaxMs: Double,
      detailNum: Int
  ): TensorLogger = {
    TensorTrace.enable()
    TensorOptionsTrace.enable()
    @volatile
    var flag = true
    val t = new Thread {
      override def run = {
        while (flag) {
          logTensorOptions(logger, filter, detailMinMs, detailMaxMs, detailNum)
          logTensors(logger, filter, detailMinMs, detailMaxMs, detailNum)
          Thread.sleep(frequency.toMillis)
        }
      }
    }
    t.start()
    val stop = () => {
      logTensorOptions(logger, filter, detailMinMs, detailMaxMs, detailNum)
      logTensors(logger, filter, detailMinMs, detailMaxMs, detailNum)
      flag = false
    }
    TensorLogger(stop)

  }

}

/** Class holding a lambda to stop the logging. See its companion object. See
  * [[lamp.data.TensorLogger#start]]
  */
case class TensorLogger(
    stop: () => Unit
) {
  def cancel() = {
    stop()
    TensorTrace.disable()
    TensorOptionsTrace.disable()
  }

}
