resolvers in ThisBuild += Resolver.githubPackages("pityka")

githubTokenSource := TokenSource.GitConfig("github.token") || TokenSource
  .Environment("GITHUB_TOKEN")

lazy val commonSettings = Seq(
  scalaVersion := "2.12.12",
  parallelExecution in Test := false,
  scalacOptions ++= Seq(
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
    "-Xlint:by-name-right-associative", // By-name parameter of right associative operator.
    "-Xlint:constant", // Evaluation of a constant arithmetic expression results in an error.
    "-Xlint:delayedinit-select", // Selecting member of DelayedInit.
    "-Xlint:doc-detached", // A Scaladoc comment appears to be detached from its element.
    "-Xlint:inaccessible", // Warn about inaccessible types in method signatures.
    "-Xlint:infer-any", // Warn when a type argument is inferred to be `Any`.
    "-Xlint:missing-interpolator", // A string literal appears to be missing an interpolator id.
    "-Xlint:nullary-override", // Warn when non-nullary `def f()' overrides nullary `def f'.
    "-Xlint:nullary-unit", // Warn when nullary methods return Unit.
    "-Xlint:option-implicit", // Option.apply used implicit view.
    "-Xlint:poly-implicit-overload", // Parameterized overloaded implicit methods are not visible as view bounds.
    "-Xlint:private-shadow", // A private field (or class parameter) shadows a superclass field.
    "-Xlint:stars-align", // Pattern sequence wildcard must align with sequence component.
    "-Xlint:type-parameter-shadow", // A local type parameter shadows a type already in scope.
    "-Xlint:unsound-match", // Pattern match may not be typesafe.
    "-Yno-adapted-args", // Do not adapt an argument list (either by inserting () or creating a tuple) to match the receiver.
    "-Ypartial-unification", // Enable partial unification in type constructor inference
    // "-Ywarn-dead-code", // Warn when dead code is identified.
    "-Ywarn-extra-implicit", // Warn when more than one implicit parameter section is defined.
    "-Ywarn-inaccessible", // Warn about inaccessible types in method signatures.
    "-Ywarn-infer-any", // Warn when a type argument is inferred to be `Any`.
    "-Ywarn-nullary-override", // Warn when non-nullary `def f()' overrides nullary `def f'.
    "-Ywarn-nullary-unit" // Warn when nullary methods return Unit.
    // "-Ywarn-numeric-widen", // Warn when numerics are widened.
    // "-Ywarn-unused:implicits", // Warn if an implicit parameter is unused.
    // "-Ywarn-unused:imports", // Warn if an import selector is not referenced.
    // "-Ywarn-unused:locals", // Warn if a local definition is unused.
    // "-Ywarn-unused:params", // Warn if a value parameter is unused.
    // "-Ywarn-unused:patvars", // Warn if a variable bound in a pattern is unused.
    // "-Ywarn-unused:privates" // Warn if a private member is unused.
  ),
  scalacOptions in (Compile, console) ~= (_ filterNot (_ == "-Xfatal-warnings"))
) ++ Seq(
  organization := "io.github.pityka",
  licenses += ("MIT", url("https://opensource.org/licenses/MIT")),
  pomExtra in Global := {
    <url>https://github.com/pityka/lamp</url>
      <developers>
        <developer>
          <id>pityka</id>
          <name>Istvan Bartha</name>
        </developer>
      </developers>
  },
  fork := true,
  cancelable in Global := true,
  githubTokenSource := TokenSource.GitConfig("github.token") || TokenSource
    .Environment("GITHUB_TOKEN"),
  coverageExcludedPackages := "lamp.example.*",
  githubOwner := "pityka",
  githubRepository := "lamp",
  publishArtifact in (Compile, packageDoc) := false,
  publishArtifact in (Compile, packageSrc) := false
)

lazy val Cuda = config("cuda").extend(Test)
lazy val AllTest = config("alltest").extend(Test)

val saddleVersion = "2.0.0-M29"
val upickleVersion = "1.2.0"

lazy val core = project
  .in(file("lamp-core"))
  .configs(Cuda)
  .configs(AllTest)
  .settings(commonSettings: _*)
  .settings(
    name := "lamp-core",
    libraryDependencies ++= Seq(
      "io.github.pityka" %% "aten-scala-core" % "0.0.0+51-7bdc55cb",
      "io.github.pityka" %% "saddle-core" % saddleVersion,
      "io.github.pityka" %% "saddle-linalg" % saddleVersion % "test",
      "org.typelevel" %% "cats-core" % "2.1.1",
      "org.typelevel" %% "cats-effect" % "2.1.3",
      "org.scalatest" %% "scalatest" % "3.1.2" % "test"
    ),
    inConfig(Cuda)(Defaults.testTasks),
    inConfig(AllTest)(Defaults.testTasks),
    testOptions in Test += Tests.Argument("-l", "cuda slow"),
    testOptions in Cuda := List(Tests.Argument("-n", "cuda")),
    testOptions in AllTest := Nil
  )

lazy val data = project
  .in(file("lamp-data"))
  .configs(Cuda)
  .configs(AllTest)
  .settings(commonSettings: _*)
  .settings(
    name := "lamp-data",
    libraryDependencies ++= Seq(
      "com.outr" %% "scribe" % "2.7.3",
      "com.lihaoyi" %% "ujson" % "1.2.0",
      "org.scalatest" %% "scalatest" % "3.1.2" % "test"
    ),
    inConfig(Cuda)(Defaults.testTasks),
    inConfig(AllTest)(Defaults.testTasks),
    testOptions in Test += Tests.Argument("-l", "cuda slow"),
    testOptions in Cuda := List(Tests.Argument("-n", "cuda")),
    testOptions in AllTest := Nil
  )
  .dependsOn(core % "test->test;compile->compile")

lazy val e2etest = project
  .in(file("endtoendtest"))
  .configs(Cuda)
  .configs(AllTest)
  .settings(commonSettings: _*)
  .settings(
    name := "lamp-e2etest",
    libraryDependencies ++= Seq(
      "org.scalatest" %% "scalatest" % "3.1.2" % "test"
    ),
    skip in publish := true,
    publishArtifact := false,
    inConfig(Cuda)(Defaults.testTasks),
    inConfig(AllTest)(Defaults.testTasks),
    testOptions in Test += Tests.Argument("-l", "cuda slow"),
    testOptions in Cuda := List(Tests.Argument("-n", "cuda")),
    testOptions in AllTest := Nil
  )
  .dependsOn(data)
  .dependsOn(forest)
  .dependsOn(core % "test->test;compile->compile")

lazy val tabular = project
  .in(file("lamp-tabular"))
  .configs(Cuda)
  .configs(AllTest)
  .settings(commonSettings: _*)
  .settings(
    name := "lamp-tabular",
    libraryDependencies ++= Seq(
      "com.lihaoyi" %% "upickle" % upickleVersion,
      "org.scalatest" %% "scalatest" % "3.1.2" % "test"
    ),
    inConfig(Cuda)(Defaults.testTasks),
    inConfig(AllTest)(Defaults.testTasks),
    testOptions in Test += Tests.Argument("-l", "cuda slow"),
    testOptions in Cuda := List(Tests.Argument("-n", "cuda")),
    testOptions in AllTest := Nil
  )
  .dependsOn(data, knn, forest)
  .dependsOn(core % "test->test;compile->compile")

lazy val umap = project
  .in(file("lamp-umap"))
  .configs(Cuda)
  .configs(AllTest)
  .settings(commonSettings: _*)
  .settings(
    name := "lamp-umap",
    libraryDependencies ++= Seq(
      "org.scalatest" %% "scalatest" % "3.1.2" % "test",
      "io.github.pityka" %% "saddle-linalg" % saddleVersion,
      "io.github.pityka" %% "nspl-awt" % "0.0.22" % "test",
      "io.github.pityka" %% "nspl-saddle" % "0.0.22" % "test"
    ),
    inConfig(Cuda)(Defaults.testTasks),
    inConfig(AllTest)(Defaults.testTasks),
    testOptions in Test += Tests.Argument("-l", "cuda slow"),
    testOptions in Cuda := List(Tests.Argument("-n", "cuda")),
    testOptions in AllTest := Nil
  )
  .dependsOn(data, knn)
  .dependsOn(core % "test->test;compile->compile")

lazy val forest = project
  .in(file("extratrees"))
  .settings(commonSettings: _*)
  .settings(
    name := "extratrees",
    libraryDependencies ++= Seq(
      "com.lihaoyi" %% "upickle" % upickleVersion,
      "org.scalatest" %% "scalatest" % "3.1.2" % "test",
      "org.typelevel" %% "cats-effect" % "2.1.3",
      "io.github.pityka" %% "saddle-linalg" % saddleVersion
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
      "org.scalatest" %% "scalatest" % "3.1.2" % "test",
      "io.github.pityka" %% "saddle-linalg" % saddleVersion
    ),
    inConfig(Cuda)(Defaults.testTasks),
    inConfig(AllTest)(Defaults.testTasks),
    testOptions in Test += Tests.Argument("-l", "cuda slow"),
    testOptions in Cuda := List(Tests.Argument("-n", "cuda")),
    testOptions in AllTest := Nil
  )
  .dependsOn(core)
  .dependsOn(core % "test->test;compile->compile")

lazy val example_cifar100 = project
  .in(file("example-cifar100"))
  .settings(commonSettings: _*)
  .settings(
    publishArtifact := false,
    skip in publish := true,
    libraryDependencies ++= Seq(
      "com.github.scopt" %% "scopt" % "4.0.0-RC2",
      "com.outr" %% "scribe" % "2.7.3"
    )
  )
  .dependsOn(core, data)
lazy val example_timemachine = project
  .in(file("example-timemachine"))
  .settings(commonSettings: _*)
  .settings(
    publishArtifact := false,
    skip in publish := true,
    libraryDependencies ++= Seq(
      "com.github.scopt" %% "scopt" % "4.0.0-RC2",
      "com.outr" %% "scribe" % "2.7.3"
    )
  )
  .dependsOn(core, data)

lazy val example_translation = project
  .in(file("example-translation"))
  .settings(commonSettings: _*)
  .settings(
    publishArtifact := false,
    skip in publish := true,
    libraryDependencies ++= Seq(
      "com.github.scopt" %% "scopt" % "4.0.0-RC2",
      "com.outr" %% "scribe" % "2.7.3"
    )
  )
  .dependsOn(core, data)

lazy val docs = project
  .in(file("lamp-docs"))
  .dependsOn(core % "compile->test;compile->compile", data)
  .settings(commonSettings: _*)
  .settings(
    publishArtifact := false,
    moduleName := "lamp-docs",
    mdocVariables := Map(
      "VERSION" -> version.value
    ),
    target in (ScalaUnidoc, unidoc) := (baseDirectory in LocalRootProject).value / "website" / "static" / "api",
    cleanFiles += (target in (ScalaUnidoc, unidoc)).value
  )
  .enablePlugins(MdocPlugin, ScalaUnidocPlugin)

lazy val root = project
  .in(file("."))
  .settings(
    publishArtifact := false,
    skip in publish := true
  )
  .aggregate(
    core,
    data,
    tabular,
    knn,
    forest,
    umap,
    docs,
    example_cifar100,
    example_timemachine,
    example_translation
  )
