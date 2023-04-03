package lamp.example.arxiv

import org.saddle._

object OgbArxivDataset {

  def unzip(zipPath: os.Path, outputPath: os.Path): Unit = {
    import java.util.zip.ZipFile
    import scala.jdk.CollectionConverters._
    val zipFile = new ZipFile(zipPath.toIO)
    for (entry <- zipFile.entries().asScala) {
      val path = outputPath.toNIO.resolve(entry.getName)
      if (entry.isDirectory) {
        java.nio.file.Files.createDirectories(path)
      } else {
        java.nio.file.Files.createDirectories(path.getParent)
        java.nio.file.Files.copy(zipFile.getInputStream(entry), path)
      }
    }
    zipFile.close()

  }

  val url = "https://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip"

  def download(folder: java.io.File) = {
    val folder1 = os.Path(folder.getAbsoluteFile)
    val edge = folder1 / "arxiv" / "raw" / "edge.csv.gz"
    val nodeYear = folder1 / "arxiv" / "raw" / "node_year.csv.gz"
    val nodeFeat = folder1 / "arxiv" / "raw" / "node-feat.csv.gz"
    val nodeLabel = folder1 / "arxiv" / "raw" / "node-label.csv.gz"
    if (
      os.exists(edge) && os.exists(nodeYear) && os.exists(nodeFeat) && os
        .exists(nodeLabel)
    ) {
      scribe.info("Found csv.gz files.")
      (edge, nodeYear, nodeFeat, nodeLabel)
    } else {
      scribe.info(s"Downloading $url")
      val arxivzip = folder1 / "arxiv.zip"
      os.write(
        arxivzip,
        requests.get.stream(url)
      )
      scribe.info(s"Unzipping $arxivzip")
      unzip(arxivzip, folder1)
      (edge, nodeYear, nodeFeat, nodeLabel)
    }

  }

  def readAll(folder: java.io.File) = {
    val (edge, nodeYear, nodeFeat, nodeLabel) = download(folder)

    (
      readAndConvert[Int](edge),
      readAndConvert[Int](nodeYear),
      readAndConvert[Float](nodeFeat),
      readAndConvert[Int](nodeLabel)
    )
  }

  def readAndConvert[T: ST](file: os.Path) = {
    val bin = os.Path(file.toIO.getAbsolutePath + ".saddle")
    if (os.exists(bin)) {
      scribe.info(s"Found $bin")
      val channel = os.read.channel(bin)
      val f = org.saddle.binary.Reader.readFrameFromChannel[T](channel)
      channel.close
      f.toOption.get
    } else {
      val frame = read[T](file)
      scribe.info(s"Convert to binary $bin")
      val channel = os.write.channel(bin)
      org.saddle.binary.Writer.writeFrameIntoChannel(frame, channel)
      channel.close
      frame
    }
  }

  def read[T: ST](file: os.Path) = {
    scribe.info(s"Parsing $file")
    val is =
      new java.util.zip.GZIPInputStream(new java.io.FileInputStream(file.toIO))
    val f =
      org.saddle.csv.CsvParser
        .parseInputStream[T](
          is,
          recordSeparator = "\n"
        )
    is.close
    f.toOption.get

  }

}
