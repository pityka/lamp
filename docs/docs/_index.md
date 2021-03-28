---
title: 'Lamp Documentation'
weight: 1
---

# Lamp

[![codecov](https://codecov.io/gh/pityka/lamp/branch/master/graph/badge.svg)](https://codecov.io/gh/pityka/lamp)
[![master](https://github.com/pityka/lamp/actions/workflows/master.yml/badge.svg)](https://github.com/pityka/lamp/actions/workflows/master.yml)
[![doc](https://img.shields.io/badge/api-scaladoc-green)](https://pityka.github.io/lamp/api/lamp/index.html)
[![doc](https://img.shields.io/badge/docs-green)](https://pityka.github.io/lamp)
[![maven](https://img.shields.io/maven-central/v/io.github.pityka/lamp-core_2.13.svg)](https://repo1.maven.org/maven2/io/github/pityka/lamp-core_2.13/)
![Sonatype Nexus (Snapshots)](https://img.shields.io/nexus/s/io.github.pityka/lamp-core_2.13?label=sonatype-snapshots&server=https%3A%2F%2Foss.sonatype.org)

Lamp is a deep learning library for Scala with native CPU and GPU backend. 

Lamp is inspired by [pytorch](https://pytorch.org/). 
The foundation of lamp is a [JNI binding to ATen](https://github.com/pityka/aten-scala), the C++ tensor backend of torch ([see here](https://pytorch.org/cppdocs/#aten])).
As a consequence lamp uses fast CPU and GPU code and stores its data in off-heap memory.

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
