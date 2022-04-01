---
title: 'Getting started'
weight: 1
---

A minimal sbt project to use lamp:

```scala
libraryDependencies += "io.github.pityka" %% "lamp-data" % "VERSION" // look at the github page for version
```

### Dependencies
- `lamp-core` depends on [cats-effect](https://github.com/typelevel/cats-effect) and [aten-scala](https://github.com/pityka/aten-scala)
- `lamp-data` in addition depends on [scribe](https://github.com/outr/scribe) and [ujson](https://github.com/lihaoyi/upickle)

Lamp depends on aten-scala is a JNI binding to libtorch. It has has cross compiled artifacts for Mac and Linux. Mac has no GPU support. Your system has to have the libtorch 1.8.0 shared libraries in its linker path.

### Verify installation
This will allocate an identity matrix of 32-bit floats in the main memory:
```scala mdoc
aten.ATen.eye_0(2L,aten.TensorOptions.dtypeFloat)
```

The following will allocate an identity matrix of 32-bit floats in the GPU memory. It will throw an exception if GPU support is not available
```scala mdoc:compile-only
aten.ATen.eye_0(2L,aten.TensorOptions.dtypeFloat.cuda)
```