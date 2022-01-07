addSbtPlugin("com.github.sbt" % "sbt-ci-release" % "1.5.10")

addSbtPlugin("com.github.sbt" % "sbt-unidoc" % "0.5.0")

addSbtPlugin("org.scalameta" % "sbt-mdoc" % "2.2.24")

addSbtPlugin("com.dwijnand" % "sbt-dynver" % "4.1.1")

addSbtPlugin("com.thesamet" % "sbt-protoc" % "1.0.5")

libraryDependencies += "com.thesamet.scalapb" %% "compilerplugin" % "0.11.8"

addSbtPlugin("com.github.sbt" % "sbt-native-packager" % "1.9.7")
