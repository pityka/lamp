package lamp

import org.saddle._
import org.saddle.order._
import org.saddle.scalar.ScalarTagDouble
import lamp.tabular.Metadata
import lamp.tabular.Numerical

sealed trait StringMetadata
object StringMetadata {

  private case class NumericNormal(
      mean: Double,
      std: Double,
      min: Double,
      max: Double,
      indicatorFor: Seq[Double],
      missingCount: Int
  ) extends StringMetadata {
    assert(std > 0d, "std == 0")
  }

  private case class Categorical(
      numericCodes: Seq[(String, Double)],
      missingCode: Option[String]
  ) extends StringMetadata
  case class Unknown(distinctSize: Int) extends StringMetadata

  private def inferNumeric(v: Vec[Double], missingCount: Int) = {
    val mean = v.mean2
    val std = v.sampleStandardDeviation
    val highFrequencyItems = {
      val idx = Index(v.toArray)
      val uniques = idx.uniques
      val freqs = idx.counts.toVec.map(x => x / v.length.toDouble)
      Series(freqs, uniques).sorted.reversed
        .head(5)
        .toSeq
        .filter(_._2 > 0.2)
        .map(_._1)
    }
    NumericNormal(mean, std, v.min2, v.max2, highFrequencyItems, missingCount)

  }

  private def inferFromColumn(
      column: Vec[String],
      numericNaN: Set[String],
      categoricalThreshold: Int,
      categoricalMissing: String
  ) = {
    val distinct = column.toArray.distinct
    val distinctSize = distinct.size
    val allNumeric = distinct.forall(v =>
      try {
        if (numericNaN.contains(v.toLowerCase)) true
        else v.toDouble
        true
      } catch {
        case _: NumberFormatException => false
      }
    )
    if (allNumeric && distinctSize > categoricalThreshold) {
      val vec1 = column.map(ScalarTagDouble.parse)
      val vec = vec1.dropNA
      inferNumeric(vec, missingCount = vec1.length - vec.length)
    } else if (distinctSize < categoricalThreshold) {
      val hasMissing = distinct.contains(categoricalMissing)
      val levelSizes = column.toArray
        .groupBy(identity)
        .toSeq
        .map(v => (v._1, v._2.size))
        .sortBy(_._2)

      val minimumLevelSize = 5

      val smallestEligibleLevel =
        levelSizes.filter(_._2 >= minimumLevelSize).size - 1

      val levels = levelSizes.reverse.zipWithIndex
        .map {
          case ((asString, size), asInt) =>
            (
              asString,
              if (size >= minimumLevelSize) asInt.toDouble
              else smallestEligibleLevel.toDouble
            )
        }
      Categorical(
        levels,
        if (hasMissing) Some(categoricalMissing) else None
      )
    } else Unknown(distinctSize)
  }

  def inferMetaFromFrame[R, C](
      frame: Frame[R, C, String],
      numericNaN: Set[String] = Set("na", "nan"),
      categoricalThreshold: Int = 100,
      categoricalMissing: String = "NA"
  ): Seq[(C, StringMetadata)] = {
    frame.toColSeq.map {
      case (c, col) =>
        (
          c,
          inferFromColumn(
            col.toVec,
            numericNaN,
            categoricalThreshold,
            categoricalMissing
          )
        )
    }
  }

  private def convertCategorical(
      column: Vec[String],
      levels: Seq[(String, Double)],
      missingLevel: Option[String]
  ) = {
    val map = levels.toMap
    List(
      column
        .map(v =>
          map.get(v).getOrElse {
            missingLevel match {
              case None              => 0d
              case Some(missingCode) => map(missingCode)
            }
          }
        ) -> lamp.tabular
        .Categorical(levels.size)
    )
  }

  private def convertNumericNormalColumn(
      column: Vec[String],
      mean: Double,
      std: Double,
      min: Double,
      max: Double,
      indicators: Seq[Double],
      missingCount: Int
  ) = {
    val v = column.map(ScalarTagDouble.parse)

    val numericValues = (v
      .map { x =>
        val v = math.max(math.min(x, max), min)
        (v - mean) / std
      }
      .fillNA(_ => mean) -> lamp.tabular.Numerical)

    val missingIndicator =
      if (missingCount >= 5)
        Some(
          (
            v.map(_ => 0d).fillNA(_ => 1d),
            lamp.tabular.Categorical(2)
          )
        )
      else None

    List(
      numericValues
    ) ++ missingIndicator.toList ++ indicators.map { i =>
      (
        v.map(v => if (v == i) 1d else 0d).fillNA(_ => 0d),
        lamp.tabular.Categorical(2)
      )
    }
  }

  private def oneHot1(v: Vec[Double], classes: Int) = {
    val zeros = 0 until classes - 1 map (_ => vec.zeros(v.length)) toArray
    var i = 0
    val n = v.length
    while (i < n) {
      val level = v.raw(i).toInt
      if (level > 0) {
        zeros(level - 1)(i) = 1d
      }
      i += 1
    }
    zeros.toList.map(v => (v, Numerical))
  }

  private def oneHot(
      columns: Seq[(Vec[Double], Metadata)],
      oneHotThreshold: Int
  ) =
    columns.flatMap {
      case (column, meta) =>
        meta match {
          case lamp.tabular.Numerical => List((column, meta))
          case lamp.tabular.Categorical(classes) =>
            if (classes > oneHotThreshold) List((column, meta))
            else {
              oneHot1(column, classes)
            }
        }
    }

  def convertFrameToTensor[R, C](
      frame: Frame[R, C, String],
      metadata: Seq[StringMetadata],
      device: Device,
      precision: FloatingPointPrecision,
      oneHotThreshold: Int
  )(implicit scope: Scope): (STen, Seq[Metadata]) = Scope { implicit scope =>
    assert(frame.numCols == metadata.size)

    val mappedColumnsWithMeta2 =
      frame.toColSeq.map(_._2.toVec).zip(metadata).flatMap {
        case (
            column,
            NumericNormal(mean, std, min, max, indicators, missingCount)
            ) =>
          convertNumericNormalColumn(
            column,
            mean,
            std,
            min,
            max,
            indicators,
            missingCount
          )
        case (_, Unknown(_)) => Nil
        case (column, Categorical(levels, missingLevel)) =>
          convertCategorical(column, levels, missingLevel)
      }
    val (cols, meta) = oneHot(mappedColumnsWithMeta2, oneHotThreshold).unzip

    val mat = Frame(cols: _*).toMat
    assert(cols.forall(v => !v.hasNA), "NA encountered")
    val tensor = STen.fromMat(mat, device, precision)
    (tensor, meta)

  }
}
