# Lamp

[![codecov](https://codecov.io/gh/pityka/lamp/branch/master/graph/badge.svg)](https://codecov.io/gh/pityka/lamp)
[![master](https://github.com/pityka/lamp/actions/workflows/master.yml/badge.svg)](https://github.com/pityka/lamp/actions/workflows/master.yml)
[![doc](https://img.shields.io/badge/api-scaladoc-green)](https://pityka.github.io/lamp/api/lamp/index.html)
[![doc](https://img.shields.io/badge/docs-green)](https://pityka.github.io/lamp)
[![maven](https://img.shields.io/maven-central/v/io.github.pityka/lamp-core_2.13.svg)](https://repo1.maven.org/maven2/io/github/pityka/lamp-core_2.13/)
[![Sonatype Nexus (Snapshots)](https://img.shields.io/nexus/s/io.github.pityka/lamp-core_2.13?label=sonatype-snapshots&server=https%3A%2F%2Foss.sonatype.org)](https://oss.sonatype.org/content/repositories/snapshots/io/github/pityka/lamp-core_2.13/)

Lamp is a Scala library for deep learning and scientific computing. 
It features a native CPU and GPU backend and operates on off-heap memory. 

Lamp is inspired by [pytorch](https://pytorch.org/). 
The foundation of lamp is a [JNI binding to ATen](https://github.com/pityka/aten-scala), the C++ tensor backend of pytorch ([see here](https://pytorch.org/cppdocs/#aten])).
As a consequence lamp uses fast CPU and GPU code and stores its data in off-heap memory.

[Documentation](https://pityka.github.io/lamp)

[API](https://pityka.github.io/lamp/api/lamp/index.html)

# Features

Lamp provides CPU or GPU backed n-dimensional arrays and implements generic automatic reverse mode differentiation (also known as autograd, see e.g. [this paper](https://arxiv.org/pdf/1811.05031.pdf)). 
Lamp may be used for scientific computing similarly to numpy, or to build neural networks.

It provides neural networks components:

- fully connected, 1D and 2D convolutional, embedding, graph convolution, self-attention (transformer), BERT
- various nonlinearities
- batch, weight, layer normalization
- dropout, weight decay
- optimizers: SgdW, AdamW (see [here](https://arxiv.org/abs/1711.05101)), RAdam, Yogi
- multi gpu data parallel training loop and data loaders
- checkpointing, ONNX export, NPY and CSV import

This repository also hosts some other loosely related libraries. 

- a fast GPU compatible implementation of UMAP ([see](https://arxiv.org/abs/1802.03426))
- an implementation of extratrees ([see](https://hal.archives-ouvertes.fr/hal-00341932)). This is a JVM implementation with no further dependencies.

# Platforms

Lamp depends on the JNI bindings in [aten-scala](https://github.com/pityka/aten-scala) which has cross compiled artifacts for Mac and Linux. Mac has no GPU support. Your system has to have the libtorch 1.9.0 shared libraries in its linker path.

On mac it suffices to copy the shared libraries from `https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.9.0.zip` to e.g. `/usr/local/lib/`.
On linux, see the following [Dockerfile](https://github.com/pityka/aten-scala/blob/master/docker-runtime/Dockerfile).

# Dependencies

In addition to the libtorch shared libraries:
- `lamp-core` depends on [saddle-core](https://github.com/pityka/saddle), [cats-effect](https://github.com/typelevel/cats-effect) and [aten-scala](https://github.com/pityka/aten-scala)
- `lamp-data` further depends on [scribe](https://github.com/outr/scribe) and [jsoniter-scala](https://github.com/plokhotnyuk/jsoniter-scala)

# Completeness

The machine generated ATen JNI binding ([aten-scala](https://github.com/pityka/aten-scala)) exposes hundreds of tensor operations from libtorch. 
On top of those tensors lamp provides autograd for the operations needed to build neural networks.

# Correctness

There is substantial test coverage in terms of unit tests and a suite of end to end tests which compares lamp to PyTorch on 50 datasets. All gradient operations and neural network modules are tested for correctness using numeric differentiation, both on CPU and GPU. Nevertheless, advance with caution.

# Getting started

Add to build.sbt:

```scala
libraryDependencies += "io.github.pityka" %% "lamp-data" % "VERSION" // look at the github page for version
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

# Building from source

First, one has to build the JNI binding to libtorch, then build lamp itself.

## Building the JNI binding

The JNI binding is hosted in the [pityka/aten-scala](https://github.com/pityka/aten-scala) git repository.
Refer to the readme in that repository on how to build the JNI sources and publish them as a scala library.

## Building lamp locally

Lamp itself is a pure Scala library and builds like any other Scala project. 
Once `aten-scala` is published to a local repository invoking `sbt compile` will work.
If you modified the package name, version or organization in the `aten-scala` build, then you have to adjust the build definition of lamp.

# License

See the LICENSE file. Licensed under the MIT License.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
