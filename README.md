# Lamp

[![codecov](https://codecov.io/gh/pityka/lamp/branch/master/graph/badge.svg)](https://codecov.io/gh/pityka/lamp)
[![](https://github.com/pityka/lamp/workflows/CI/badge.svg)](https://github.com/pityka/lamp/actions?query=workflow%3ACI)

Lamp is a deep learning library for Scala with native CPU and GPU backend. 

Lamp is inspired by [pytorch](https://pytorch.org/). 
The foundation of lamp is a [JNI binding to ATen](https://github.com/pityka/aten-scala), the C++ tensor backend of torch ([see here](https://pytorch.org/cppdocs/#aten])).
As a consequence lamp uses fast CPU and GPU code and stores its data in off-heap memory.

# Features

Lamp implements generic automatic reverse mode differentiation (also known as autograd, see e.g. [this paper](https://arxiv.org/pdf/1811.05031.pdf)). 

On top of that it provides a small set of components to build neural networks:

- fully connected, 1D and 2D convolutional, RNN and GRU layers
- relu and gelu nonlinearities
- batch normalization and weight normalization
- dropout
- SgdW and AdamW optimizers (see [here](https://arxiv.org/abs/1711.05101))
- training loop and data loaders on top of cats-effect
- checkpointing

All gradient operations are tested for correctness with numeric differentiation.
All of these tests are replicated to the GPU as well.

# Platforms

Lamp depends on the JNI bindings in [aten-scala](https://github.com/pityka/aten-scala) which has cross compiled artifacts for Mac and Linux. Mac has no GPU support. Your system has to have the libtorch 1.5.0 shared libraries in its linker path.

On mac it suffices to install torch with `brew install libtorch`.
On linux, see the following [Dockerfile](https://github.com/pityka/aten-scala/blob/master/docker-runtime/Dockerfile).

# Getting started

Lamp is experimental, and no artifacts are pubished to maven central.

The aten-scala artifacts are published to Github Packages, which needs a github user token available either in a $GITHUB_TOKEN environmental variable, or in the git global configuration (`~/.gitconfig`): 
```gitconfig
[github]
  token = TOKEN_DATA
```
The aten-scala artifacts are public, nevertheless Github still requires authentication.

## Running tests

`sbt test`

Cuda tests are run separately with `sbt cuda:test`. See `test_cuda.sh` in the source tree about how to run this in a remote docker context.

## Running the cifar-100 example

See `run_cifar.sh` in the source tree.

## Running the text model example

See `run_timemachine.sh` in the source tree.

# License

See the LICENSE file. Licensed under the MIT License.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
