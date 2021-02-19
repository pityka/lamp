# Lamp

[![codecov](https://codecov.io/gh/pityka/lamp/branch/master/graph/badge.svg)](https://codecov.io/gh/pityka/lamp)
[![](https://github.com/pityka/lamp/workflows/CI/badge.svg)](https://github.com/pityka/lamp/actions?query=workflow%3ACI)
[![doc](https://img.shields.io/badge/api-scaladoc-green)](https://pityka.github.io/lamp/api/lamp/index.html)
[![doc](https://img.shields.io/badge/docs-green)](https://pityka.github.io/lamp)

Lamp is a Scala library for deep learning and scientific computing. 
It features a native CPU and GPU backend and operates on off-heap memory. 

Lamp is inspired by [pytorch](https://pytorch.org/). 
The foundation of lamp is a [JNI binding to ATen](https://github.com/pityka/aten-scala), the C++ tensor backend of torch ([see here](https://pytorch.org/cppdocs/#aten])).
As a consequence lamp uses fast CPU and GPU code and stores its data in off-heap memory.

[Documentation](https://pityka.github.io/lamp)

[API](https://pityka.github.io/lamp/api/lamp/index.html)

# Features

Lamp provides CPU or GPU backed n-dimensional arrays and implements generic automatic reverse mode differentiation (also known as autograd, see e.g. [this paper](https://arxiv.org/pdf/1811.05031.pdf)). 
Lamp may be used for scientific computing similarly to numpy, or to build neural networks.

It provides neural networks components:

- fully connected, 1D and 2D convolutional, embedding, RNN, GRU, LSTM, GCN, self-attention (transformer) layers
- various nonlinearities
- batch normalization and weight normalization
- seq2seq
- dropout
- optimizers: SgdW, AdamW (see [here](https://arxiv.org/abs/1711.05101)), RAdam, Yogi
- training loop and data loaders on top of cats-effect
- checkpointing, ONNX export, NPY and CSV import

This repository also hosts some other loosely related libraries. 

- a fast GPU compatible implementation of UMAP ([see](https://arxiv.org/abs/1802.03426))
- an implementation of extratrees ([see](https://hal.archives-ouvertes.fr/hal-00341932)). This is a JVM implementation with no further dependencies.

# Platforms

Lamp depends on the JNI bindings in [aten-scala](https://github.com/pityka/aten-scala) which has cross compiled artifacts for Mac and Linux. Mac has no GPU support. Your system has to have the libtorch 1.7.1 shared libraries in its linker path.

On mac it suffices to copy the shared libraries from `https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.7.1.zip` to e.g. `/usr/local/lib/`.
On linux, see the following [Dockerfile](https://github.com/pityka/aten-scala/blob/master/docker-runtime/Dockerfile).

# Dependencies

In addition to the libtorch shared libraries:
- `lamp-core` depends on [saddle-core](https://github.com/pityka/saddle), [cats-effect](https://github.com/typelevel/cats-effect) and [aten-scala](https://github.com/pityka/aten-scala)
- `lamp-data` further depends on [scribe](https://github.com/outr/scribe) and [ujson](https://github.com/lihaoyi/upickle)

# Completeness

The machine generated ATen JNI binding ([aten-scala](https://github.com/pityka/aten-scala)) exposes hundreds of tensor operations from libtorch. 
On top of those tensors lamp provides autograd for the operations needed to build neural networks.

# Correctness

There is substantial test coverage in terms of unit tests and a suite of end to end tests which compares lamp to PyTorch on 50 datasets. All gradient operations and neural network modules are tested for correctness using numeric differentiation, both on CPU and GPU. Nevertheless, advance with caution.

# Getting started

Lamp is experimental, and no artifacts are pubished to maven central. Artifacts of lamp and aten-scala are delivered to Github Packages. Despite the artifacts being public, you need to authenticate to Github.

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


## Running tests

`sbt test` will run a short test suite of unit tests.

Cuda tests are run separately with `sbt cuda:test`. See `test_cuda.sh` in the source tree about how to run this in a remote docker context. Some additional tests are run from `test_slow.sh`.

All tests are executed with `sbt alltest:test`. This runs all unit tests, all cuda tests, additional tests marked as slow, and a more extensive end-to-end benchmark against PyTorch itself on 50 datasets.

## Examples

Examples for various tasks:

- Image classification: `bash run_cifar.sh` runs the code in `example-cifar100/`.
- Text generation: `bash run_timemachine.sh` runs the code in `example-timemachine/`.
- Machine translation: `bash run_translation.sh` runs the code in `example-translation/`.
- Graph node property prediction: `bash run_arxiv.sh` runs the code in `example-arxiv/`.


# License

See the LICENSE file. Licensed under the MIT License.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
