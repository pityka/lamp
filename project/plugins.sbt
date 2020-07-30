addSbtPlugin("com.codecommit" % "sbt-github-packages" % "0.5.0")

resolvers += Resolver.bintrayRepo("djspiewak", "maven")

addSbtPlugin("org.scoverage" % "sbt-scoverage" % "1.6.1")

addSbtPlugin("com.eed3si9n" % "sbt-unidoc" % "0.4.3")

addSbtPlugin("org.scalameta" % "sbt-mdoc" % "2.2.1")

addSbtPlugin("com.geirsson" % "sbt-ci-release" % "1.5.3")
