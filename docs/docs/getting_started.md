---
title: 'Getting started'
weight: 1
---

Lamp is experimental, and no artifacts are pubished to maven central.

### Build and publish into local repository

The aten-scala artifacts are published to Github Packages, which needs a github user token available either in a $GITHUB_TOKEN environmental variable, or in the git global configuration (`~/.gitconfig`): 
```gitconfig
[github]
  token = TOKEN_DATA
```

Publish into local repository with `sbt pubishLocal`.

### Dependencies
- `lamp-core` depends on [saddle-core](https://github.com/pityka/saddle), [cats-effect](https://github.com/typelevel/cats-effect) and [aten-scala](https://github.com/pityka/aten-scala)
- `lamp-data` in addition depends on [scribe](https://github.com/outr/scribe) and [ujson](https://github.com/lihaoyi/upickle)

Lamp depends on aten-scala is a JNI binding to libtorch. It has has cross compiled artifacts for Mac and Linux. Mac has no GPU support. Your system has to have the libtorch 1.5.0 shared libraries in its linker path.

On mac it suffices to install torch with `brew install libtorch`.
On linux, see the following [Dockerfile](https://github.com/pityka/aten-scala/blob/master/docker-runtime/Dockerfile).

### Verify installation
This will allocate an identity matrix of 32-bit floats in the main memory:
```scala mdoc
aten.ATen.eye_0(2L,aten.TensorOptions.dtypeFloat)
```

The following will allocate an identity matrix of 32-bit floats in the GPU memory. It will throw an exception if GPU support is not available
```scala mdoc:compile-only
aten.ATen.eye_0(2L,aten.TensorOptions.dtypeFloat.cuda)
```