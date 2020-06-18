---
title: 'Lamp Documentation'
weight: 1
---

# Lamp

[![codecov](https://codecov.io/gh/pityka/lamp/branch/master/graph/badge.svg)](https://codecov.io/gh/pityka/lamp)
[![](https://github.com/pityka/lamp/workflows/CI/badge.svg)](https://github.com/pityka/lamp/actions?query=workflow%3ACI)
[![doc](https://img.shields.io/badge/api-scaladoc-green)](https://pityka.github.io/lamp/api/lamp/index.html)
[![doc](https://img.shields.io/badge/docs-green)](https://pityka.github.io/lamp)

Lamp is small a deep learning library for Scala with native CPU and GPU backend. 

Lamp is inspired by [pytorch](https://pytorch.org/). 
The foundation of lamp is a [JNI binding to ATen](https://github.com/pityka/aten-scala), the C++ tensor backend of torch ([see here](https://pytorch.org/cppdocs/#aten])).
As a consequence lamp uses fast CPU and GPU code and stores its data in off-heap memory.

# Features

Lamp implements generic automatic reverse mode differentiation (also known as autograd, see e.g. [this paper](https://arxiv.org/pdf/1811.05031.pdf)). 

On top of that it provides a small set of components to build neural networks:

- fully connected, 1D and 2D convolutional, RNN and GRU layers
- various nonlinearities
- batch normalization and weight normalization
- dropout
- SgdW and AdamW optimizers (see [here](https://arxiv.org/abs/1711.05101))
- training loop and data loaders on top of cats-effect
- checkpointing
