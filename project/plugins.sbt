addSbtPlugin("com.geirsson" % "sbt-ci-release" % "1.5.6")

addSbtPlugin("com.eed3si9n" % "sbt-unidoc" % "0.4.3")

addSbtPlugin("org.scalameta" % "sbt-mdoc" % "2.2.1")

addSbtPlugin("com.dwijnand" % "sbt-dynver" % "4.0.0")

addSbtPlugin("com.thesamet" % "sbt-protoc" % "1.0.0-RC2")

libraryDependencies += "com.thesamet.scalapb" %% "compilerplugin" % "0.10.11"
