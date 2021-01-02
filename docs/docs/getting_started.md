---
title: 'Getting started'
weight: 1
---

Artifacts of lamp and aten-scala are delivered to Github Packages. Despite the artifacts being public, you need to authenticate to Github.

A minimal sbt project to use lamp:

```scala
// in build.sbt
scalaVersion := "2.12.12"

resolvers in ThisBuild += Resolver.githubPackages("pityka")

githubTokenSource := TokenSource.GitConfig("github.token") || TokenSource
  .Environment("GITHUB_TOKEN")

libraryDependencies += "io.github.pityka" %% "lamp-data" % "VERSION" // look at the github project page for version
```

```scala
// in project/plugins.sbt
addSbtPlugin("com.codecommit" % "sbt-github-packages" % "0.5.0")

resolvers += Resolver.bintrayRepo("djspiewak", "maven")
```

```scala
// in project/build.properties
sbt.version=1.3.13
```

Github Packages needs a github user token available either in a $GITHUB_TOKEN environmental variable, or in the git global configuration (`~/.gitconfig`): 
```gitconfig
[github]
  token = TOKEN_DATA
```

### Dependencies
- `lamp-core` depends on [saddle-core](https://github.com/pityka/saddle), [cats-effect](https://github.com/typelevel/cats-effect) and [aten-scala](https://github.com/pityka/aten-scala)
- `lamp-data` in addition depends on [scribe](https://github.com/outr/scribe) and [ujson](https://github.com/lihaoyi/upickle)

Lamp depends on aten-scala is a JNI binding to libtorch. It has has cross compiled artifacts for Mac and Linux. Mac has no GPU support. Your system has to have the libtorch 1.7.1 shared libraries in its linker path.

### Verify installation
This will allocate an identity matrix of 32-bit floats in the main memory:
```scala mdoc
aten.ATen.eye_0(2L,aten.TensorOptions.dtypeFloat)
```

The following will allocate an identity matrix of 32-bit floats in the GPU memory. It will throw an exception if GPU support is not available
```scala mdoc:compile-only
aten.ATen.eye_0(2L,aten.TensorOptions.dtypeFloat.cuda)
```