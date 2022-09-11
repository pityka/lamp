inThisBuild(
  List(
    organization := "io.github.pityka",
    homepage := Some(url("https://pityka.github.io/lamp/")),
    licenses := List(("MIT", url("https://opensource.org/licenses/MIT"))),
    developers := List(
      Developer(
        "pityka",
        "Istvan Bartha",
        "bartha.pityu@gmail.com",
        url("https://github.com/pityka/lamp")
      )
    ),
    parallelExecution := false
  )
)

lazy val commonSettings = Seq(
  scalaVersion := "2.13.8",
  crossScalaVersions := Seq("2.13.8", "3.1.3"),
  Test / parallelExecution := false,
  scalacOptions ++= (CrossVersion.partialVersion(scalaVersion.value) match {
    case Some((3, _)) =>
      Seq(
        "-deprecation", // Emit warning and location for usages of deprecated APIs.
        "-encoding",
        "utf-8", // Specify character encoding used by source files.
        "-feature", // Emit warning and location for usages of features that should be imported explicitly.
        "-language:postfixOps",
        "-language:existentials",
        "-unchecked", // Enable additional warnings where generated code depends on assumptions.
        "-Xfatal-warnings" // Fail the compilation if there are any warnings.
      )
    case Some((2, _)) =>
      Seq(
        "-opt:l:method",
        "-opt:l:inline",
        "-opt-inline-from:org.saddle.**",
        "-opt-warnings",
        "-deprecation", // Emit warning and location for usages of deprecated APIs.
        "-encoding",
        "utf-8", // Specify character encoding used by source files.
        "-feature", // Emit warning and location for usages of features that should be imported explicitly.
        "-language:postfixOps",
        "-language:existentials",
        "-unchecked", // Enable additional warnings where generated code depends on assumptions.
        "-Xfatal-warnings", // Fail the compilation if there are any warnings.
        "-Xlint:adapted-args", // Warn if an argument list is modified to match the receiver.
        "-Xlint:constant", // Evaluation of a constant arithmetic expression results in an error.
        "-Xlint:delayedinit-select", // Selecting member of DelayedInit.
        "-Xlint:doc-detached", // A Scaladoc comment appears to be detached from its element.
        "-Xlint:inaccessible", // Warn about inaccessible types in method signatures.
        "-Xlint:infer-any", // Warn when a type argument is inferred to be `Any`.
        "-Xlint:missing-interpolator", // A string literal appears to be missing an interpolator id.
        "-Xlint:nullary-unit", // Warn when nullary methods return Unit.
        "-Xlint:option-implicit", // Option.apply used implicit view.
        "-Xlint:poly-implicit-overload", // Parameterized overloaded implicit methods are not visible as view bounds.
        "-Xlint:private-shadow", // A private field (or class parameter) shadows a superclass field.
        "-Xlint:stars-align", // Pattern sequence wildcard must align with sequence component.
        "-Xlint:type-parameter-shadow", // A local type parameter shadows a type already in scope.
        // "-Ywarn-dead-code", // Warn when dead code is identified.
        // "-Ywarn-numeric-widen", // Warn when numerics are widened.
        "-Ywarn-unused:implicits", // Warn if an implicit parameter is unused.
        "-Ywarn-unused:imports", // Warn if an import selector is not referenced.
        "-Ywarn-unused:locals", // Warn if a local definition is unused.
        "-Ywarn-unused:params", // Warn if a value parameter is unused.
        "-Ywarn-unused:patvars", // Warn if a variable bound in a pattern is unused.
        "-Ywarn-unused:privates" // Warn if a private member is unused.
      )
    case _ => ???
  }),
  Compile / console / scalacOptions ~= (_ filterNot (_ == "-Xfatal-warnings")),
  Compile / doc / scalacOptions ~= (_ filterNot (_ == "-Xfatal-warnings"))
) ++ Seq(
  fork := true,
  run / javaOptions += "-Xmx12G",
  cancelable in Global := true
)

lazy val Cuda = config("cuda").extend(Test)
lazy val AllTest = config("alltest").extend(Test)

val saddleVersion = "3.5.0"
val upickleVersion = "1.6.0"
val scalaTestVersion = "3.2.10"
val scribeVersion = "3.8.2"
val catsEffectVersion = "3.3.14"
val catsCoreVersion = "2.8.0"
val jsoniterscalaVersion = "2.13.36"

lazy val saddlecompat = project
  .in(file("lamp-saddle"))
  .settings(commonSettings: _*)
  .settings(
    name := "lamp-saddle",
    libraryDependencies ++= Seq(
      "io.github.pityka" %% "saddle-core" % saddleVersion,
      "io.github.pityka" %% "saddle-linalg" % saddleVersion % "test",
      "org.scalatest" %% "scalatest" % scalaTestVersion % "test"
    )
  )
  .dependsOn(sten)

lazy val akkacommunicator = project
  .in(file("lamp-akka"))
  .settings(commonSettings: _*)
  .settings(
    name := "lamp-akka",
    libraryDependencies ++= Seq(
      "com.typesafe.akka" %% "akka-actor" % "2.6.19" % Provided,
      "com.typesafe.akka" %% "akka-remote" % "2.6.19" % Provided,
      "org.scalatest" %% "scalatest" % scalaTestVersion % "test"
    )
  )
  .dependsOn(data % "compile->compile;test->test")

lazy val sten = project
  .in(file("lamp-sten"))
  .configs(Cuda)
  .configs(AllTest)
  .settings(commonSettings: _*)
  .settings(
    name := "lamp-sten",
    libraryDependencies ++= Seq(
      "io.github.pityka" %% "aten-scala-core" % "0.0.0+106-60284546",
      "org.typelevel" %% "cats-core" % catsCoreVersion,
      "org.typelevel" %% "cats-effect" % catsEffectVersion,
      "org.scalatest" %% "scalatest" % scalaTestVersion % "test"
    ),
    inConfig(Cuda)(Defaults.testTasks),
    inConfig(AllTest)(Defaults.testTasks),
    Test / testOptions += Tests.Argument("-l", "cuda slow"),
    Cuda / testOptions := List(Tests.Argument("-n", "cuda")),
    AllTest / testOptions := Nil
  )

lazy val core = project
  .in(file("lamp-core"))
  .configs(Cuda)
  .configs(AllTest)
  .settings(commonSettings: _*)
  .settings(
    name := "lamp-core",
    inConfig(Cuda)(Defaults.testTasks),
    inConfig(AllTest)(Defaults.testTasks),
    Test / testOptions += Tests.Argument("-l", "cuda slow"),
    Cuda / testOptions := List(Tests.Argument("-n", "cuda")),
    AllTest / testOptions := Nil,
    libraryDependencies ++= List(
      "io.github.pityka" %% "saddle-linalg" % saddleVersion % "test"
    )
  )
  .dependsOn(sten % "test->test;compile->compile", saddlecompat % "test->test")

lazy val data = project
  .in(file("lamp-data"))
  .configs(Cuda)
  .configs(AllTest)
  .settings(commonSettings: _*)
  .settings(
    name := "lamp-data",
    libraryDependencies ++= Seq(
      "com.outr" %% "scribe" % scribeVersion,
      "com.github.plokhotnyuk.jsoniter-scala" %% "jsoniter-scala-core" % jsoniterscalaVersion,
      "com.github.plokhotnyuk.jsoniter-scala" %% "jsoniter-scala-macros" % jsoniterscalaVersion % "compile-internal"
    ),
    inConfig(Cuda)(Defaults.testTasks),
    inConfig(AllTest)(Defaults.testTasks),
    Test / testOptions += Tests.Argument("-l", "cuda slow"),
    Cuda / testOptions := List(Tests.Argument("-n", "cuda")),
    AllTest / testOptions := Nil
  )
  .dependsOn(core % "test->test;compile->compile", onnx % "test")

lazy val e2etest = project
  .in(file("endtoendtest"))
  .configs(Cuda)
  .configs(AllTest)
  .settings(commonSettings: _*)
  .settings(
    name := "lamp-e2etest",
    libraryDependencies ++= Seq(
      "org.scalatest" %% "scalatest" % scalaTestVersion % "test"
    ),
    publish / skip := true,
    publishArtifact := false,
    inConfig(Cuda)(Defaults.testTasks),
    inConfig(AllTest)(Defaults.testTasks),
    Test / testOptions += Tests.Argument("-l", "cuda slow"),
    Cuda / testOptions := List(Tests.Argument("-n", "cuda")),
    AllTest / testOptions := Nil
  )
  .dependsOn(data)
  .dependsOn(forest, saddlecompat)
  .dependsOn(core % "test->test;compile->compile")

lazy val umap = project
  .in(file("lamp-umap"))
  .configs(Cuda)
  .configs(AllTest)
  .settings(commonSettings: _*)
  .settings(
    name := "lamp-umap",
    libraryDependencies ++= Seq(
      "org.scalatest" %% "scalatest" % scalaTestVersion % "test"
    ),
    inConfig(Cuda)(Defaults.testTasks),
    inConfig(AllTest)(Defaults.testTasks),
    Test / testOptions += Tests.Argument("-l", "cuda slow"),
    Cuda / testOptions := List(Tests.Argument("-n", "cuda")),
    AllTest / testOptions := Nil
  )
  .dependsOn(data, knn, saddlecompat % "test")
  .dependsOn(core % "test->test;compile->compile")

lazy val onnx = project
  .in(file("lamp-onnx"))
  .configs(Cuda)
  .configs(AllTest)
  .settings(commonSettings: _*)
  .settings(
    name := "lamp-onnx",
    libraryDependencies ++= Seq(
      "org.scalatest" %% "scalatest" % scalaTestVersion % "test",
      "com.thesamet.scalapb" %% "scalapb-runtime" % scalapb.compiler.Version.scalapbVersion % "protobuf",
      "com.microsoft.onnxruntime" % "onnxruntime" % "1.11.0" % "test"
    ),
    Compile / PB.targets := Seq(
      scalapb.gen() -> (Compile / sourceManaged).value / "scalapb"
    ),
    inConfig(Cuda)(Defaults.testTasks),
    inConfig(AllTest)(Defaults.testTasks),
    Test / testOptions += Tests.Argument("-l", "cuda slow"),
    Cuda / testOptions := List(Tests.Argument("-n", "cuda")),
    AllTest / testOptions := Nil
  )
  .dependsOn(core % "test->test;compile->compile")

lazy val forest = project
  .in(file("extratrees"))
  .settings(commonSettings: _*)
  .settings(
    name := "extratrees",
    libraryDependencies ++= Seq(
      "com.lihaoyi" %% "upickle" % upickleVersion,
      "org.scalatest" %% "scalatest" % scalaTestVersion % "test",
      "org.typelevel" %% "cats-effect" % catsEffectVersion,
      "io.github.pityka" %% "saddle-core" % saddleVersion
    )
  )
  .dependsOn(core % "test->test")

lazy val knn = project
  .in(file("lamp-knn"))
  .configs(Cuda)
  .configs(AllTest)
  .settings(commonSettings: _*)
  .settings(
    name := "lamp-knn",
    libraryDependencies ++= Seq(
      "org.scalatest" %% "scalatest" % scalaTestVersion % "test",
      "io.github.pityka" %% "saddle-linalg" % saddleVersion
    ),
    inConfig(Cuda)(Defaults.testTasks),
    inConfig(AllTest)(Defaults.testTasks),
    Test / testOptions += Tests.Argument("-l", "cuda slow"),
    Cuda / testOptions := List(Tests.Argument("-n", "cuda")),
    AllTest / testOptions := Nil
  )
  .dependsOn(core, saddlecompat)
  .dependsOn(core % "test->test;compile->compile")

lazy val example_cifar100 = project
  .in(file("example-cifar100"))
  .settings(commonSettings: _*)
  .settings(
    publishArtifact := false,
    publish / skip := true,
    libraryDependencies ++= Seq(
      "com.github.scopt" %% "scopt" % "4.1.0",
      "io.github.pityka" %% "saddle-core" % saddleVersion,
      "com.outr" %% "scribe" % scribeVersion
    )
  )
  .dependsOn(core, data, onnx, saddlecompat)
  .enablePlugins(JavaAppPackaging)

lazy val example_cifar100_distributed = project
  .in(file("example-cifar100-distributed"))
  .settings(commonSettings: _*)
  .settings(
    publishArtifact := false,
    publish / skip := true,
    libraryDependencies ++= Seq(
      "com.github.scopt" %% "scopt" % "4.0.1",
      "com.typesafe.akka" %% "akka-actor" % "2.6.19",
      "com.typesafe.akka" %% "akka-remote" % "2.6.19",
      "io.github.pityka" %% "saddle-core" % saddleVersion,
      "com.outr" %% "scribe" % scribeVersion
    )
  )
  .dependsOn(core, data, onnx, saddlecompat, akkacommunicator)
  .enablePlugins(JavaAppPackaging)

lazy val example_gan = project
  .in(file("example-gan"))
  .settings(commonSettings: _*)
  .settings(
    publishArtifact := false,
    publish / skip := true,
    libraryDependencies ++= Seq(
      "com.github.scopt" %% "scopt" % "4.1.0",
      "io.github.pityka" %% "saddle-core" % saddleVersion,
      "com.outr" %% "scribe" % scribeVersion
    )
  )
  .dependsOn(core, data, onnx, saddlecompat)
lazy val example_timemachine = project
  .in(file("example-timemachine"))
  .settings(commonSettings: _*)
  .settings(
    publishArtifact := false,
    publish / skip := true,
    libraryDependencies ++= Seq(
      "com.github.scopt" %% "scopt" % "4.1.0",
      "io.github.pityka" %% "saddle-core" % saddleVersion,
      "com.outr" %% "scribe" % scribeVersion
    )
  )
  .dependsOn(core, data, saddlecompat)
lazy val example_bert = project
  .in(file("example-bert"))
  .settings(commonSettings: _*)
  .settings(
    publishArtifact := false,
    publish / skip := true,
    libraryDependencies ++= Seq(
      "com.github.scopt" %% "scopt" % "4.1.0",
      "io.github.pityka" %% "saddle-core" % saddleVersion,
      "com.outr" %% "scribe" % scribeVersion
    )
  )
  .dependsOn(core, data, saddlecompat)

lazy val example_translation = project
  .in(file("example-translation"))
  .settings(commonSettings: _*)
  .settings(
    publishArtifact := false,
    publish / skip := true,
    libraryDependencies ++= Seq(
      "com.github.scopt" %% "scopt" % "4.1.0",
      "io.github.pityka" %% "saddle-core" % saddleVersion,
      "com.outr" %% "scribe" % scribeVersion
    )
  )
  .dependsOn(core, data, saddlecompat)

lazy val example_arxiv = project
  .in(file("example-arxiv"))
  .settings(commonSettings: _*)
  .settings(
    publishArtifact := false,
    publish / skip := true,
    libraryDependencies ++= Seq(
      "com.github.scopt" %% "scopt" % "4.1.0",
      "com.outr" %% "scribe" % scribeVersion,
      "io.github.pityka" %% "saddle-binary" % saddleVersion,
      "io.github.pityka" %% "saddle-core" % saddleVersion,
      "com.lihaoyi" %% "requests" % "0.6.9",
      "com.lihaoyi" %% "os-lib" % "0.8.1"
    )
  )
  .dependsOn(core, data, saddlecompat)

lazy val docs = project
  .in(file("lamp-docs"))
  .dependsOn(
    core % "compile->test;compile->compile",
    data,
    forest,
    saddlecompat
  )
  .settings(commonSettings: _*)
  .settings(
    publishArtifact := false,
    moduleName := "lamp-docs",
    mdocVariables := Map(
      "VERSION" -> version.value
    ),
    ScalaUnidoc / unidoc / target := (LocalRootProject / baseDirectory).value / "website" / "static" / "api",
    cleanFiles += (ScalaUnidoc / unidoc / target).value,
      ScalaUnidoc / unidoc / unidocProjectFilter :=
      (inAnyProject -- inProjects(
        example_arxiv,
        example_bert,
        example_cifar100,
        example_cifar100_distributed,
        example_gan,
        example_timemachine,
        example_translation,
        e2etest,
      )),
  )
  .enablePlugins(MdocPlugin, ScalaUnidocPlugin)

lazy val root = project
  .in(file("."))
  .settings(
    crossScalaVersions := Nil,
    publishArtifact := false,
    publish / skip := true
  )
  .aggregate(
    sten,
    saddlecompat,
    akkacommunicator,
    core,
    data,
    knn,
    forest,
    umap,
    onnx,
    docs,
    example_cifar100,
    example_cifar100_distributed,
    example_timemachine,
    example_translation,
    example_arxiv,
    example_gan,
    example_bert,
    e2etest
  )
