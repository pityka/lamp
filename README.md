# Lamp

[![codecov](https://codecov.io/gh/pityka/lamp/branch/master/graph/badge.svg)](https://codecov.io/gh/pityka/lamp)
[![](https://github.com/pityka/lamp/workflows/CI/badge.svg)](https://github.com/pityka/lamp/actions?query=workflow%3ACI)
[![doc](https://img.shields.io/badge/api-scaladoc-green)](https://pityka.github.io/lamp/api/lamp/index.html)
[![doc](https://img.shields.io/badge/docs-green)](https://pityka.github.io/lamp)

Lamp is a deep learning library for Scala with native CPU and GPU backend. 

Lamp is inspired by [pytorch](https://pytorch.org/). 
The foundation of lamp is a [JNI binding to ATen](https://github.com/pityka/aten-scala), the C++ tensor backend of torch ([see here](https://pytorch.org/cppdocs/#aten])).
As a consequence lamp uses fast CPU and GPU code and stores its data in off-heap memory.

# Features

Lamp implements generic automatic reverse mode differentiation (also known as autograd, see e.g. [this paper](https://arxiv.org/pdf/1811.05031.pdf)). 

On top of that it provides a small set of components to build neural networks:

- fully connected, 1D and 2D convolutional, embedding, RNN, GRU, LSTM layers
- various nonlinearities
- batch normalization and weight normalization
- seq2seq
- dropout
- SgdW and AdamW optimizers (see [here](https://arxiv.org/abs/1711.05101))
- training loop and data loaders on top of cats-effect
- checkpointing

# Platforms

Lamp depends on the JNI bindings in [aten-scala](https://github.com/pityka/aten-scala) which has cross compiled artifacts for Mac and Linux. Mac has no GPU support. Your system has to have the libtorch 1.5.0 shared libraries in its linker path.

On mac it suffices to install torch with `brew install libtorch`.
On linux, see the following [Dockerfile](https://github.com/pityka/aten-scala/blob/master/docker-runtime/Dockerfile).

# Completeness

The machine generated ATen JNI binding ([aten-scala](https://github.com/pityka/aten-scala)) exposes hundreds of tensor operations from libtorch. 
On top of those lamp provides autograd for the operations needed to build neural networks. The library is expressive enough to implement common models for text, image and tabular data processing.

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

## Running the cifar-100 example

See `run_cifar.sh` in the source tree.

## Running the text model example

See `run_timemachine.sh` in the source tree.

## Running the machine translation example

See `run_translation.sh` in the source tree.

# License

See the LICENSE file. Licensed under the MIT License.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
