resolvers in ThisBuild += Resolver.githubPackages("pityka")

githubTokenSource := TokenSource.GitConfig("github.token") || TokenSource
  .Environment("GITHUB_TOKEN")

lazy val commonSettings = Seq(
  scalaVersion := "2.12.11",
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
    <url>https://github.com/pityka/candle</url>
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
  coverageExcludedPackages := "lamp.example.*"
)

lazy val Cuda = config("cuda").extend(Test)

lazy val core = project
  .in(file("lamp-core"))
  .configs(Cuda)
  .settings(commonSettings: _*)
  .settings(
    name := "lamp-autograd",
    libraryDependencies ++= Seq(
      "io.github.pityka" %% "aten-scala-core" % "0.0.0+33-75d357c5",
      "io.github.pityka" %% "saddle-core" % "2.0.0-M25",
      "io.github.pityka" %% "saddle-linalg" % "2.0.0-M25",
      "org.typelevel" %% "cats-core" % "2.1.1",
      "org.typelevel" %% "cats-effect" % "2.1.3",
      "com.lihaoyi" %% "ujson" % "1.1.0",
      "org.scalatest" %% "scalatest" % "3.1.2" % "test"
    ),
    inConfig(Cuda)(Defaults.testTasks),
    testOptions in Test += Tests.Argument("-l", "cuda"),
    testOptions in Cuda -= Tests.Argument("-l", "cuda"),
    testOptions in Cuda += Tests.Argument("-n", "cuda")
  )

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
  .dependsOn(core)
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
  .dependsOn(core)

lazy val root = project
  .in(file("."))
  .settings(
    publishArtifact := false,
    skip in publish := true
  )
  .aggregate(core, example_cifar100)
