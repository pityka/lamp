---
title: 'How to build and train a model'
weight: 2
---

# Overview

Lamp's design closely follows the design of pytorch. It is organized into the following components:
1. A native tensor library storing multidimensional arrays of numbers in off-heap (main or GPU) memory. This is in the `aten` package. 
2. An algorithm to compute partial derivatives of complex functions. In particular Lamp implements generic reverse mode automatic differentiation. This lives in the package `lamp.autograd`.
3. A set of building blocks to build neural networks in the package `lamp.nn`.
4. Training loop and utilities to work with image and text data in `lamp.data`.

# Building a classifier

This document shows how to build and train a simple model. The following code is taken from the test suite.

We will use tabular data to build a multiclass classifier.

## Get some data

First get some tabular data into the JVM memory:

```scala mdoc
    val testData = org.saddle.csv.CsvParser
      .parseSourceWithHeader[Double](
        scala.io.Source
          .fromInputStream(
            new java.util.zip.GZIPInputStream(
              getClass.getResourceAsStream("/mnist_test.csv.gz")
            )
          ),
        maxLines = 100L
      )
      .right
      .get
    val trainData = org.saddle.csv.CsvParser
      .parseSourceWithHeader[Double](
        scala.io.Source
          .fromInputStream(
            new java.util.zip.GZIPInputStream(
              getClass.getResourceAsStream("/mnist_train.csv.gz")
            )
          ),
        maxLines = 100L
      )
      .right
      .get
```

Copy those data `aten.Tensor`s of respective shape and type. Note that we specify device so it will end up on the desired device.
The feature matrix is copied into a 2D floating point tensor and the target vector is holding the class labels is copied into a 1D integer (long) tensor.
```scala mdoc
    import lamp._
    import lamp.autograd._
    import aten.ATen
    import org.saddle.Mat
    val device = CPU
    val precision = SinglePrecision
    val testDataTensor =
      TensorHelpers.fromMat(testData.filterIx(_ != "label").toMat, device, precision)
    val testTarget = ATen.squeeze_0(
      TensorHelpers.fromLongMat(
        Mat(testData.firstCol("label").toVec.map(_.toLong)),
        device
      )
    )

    val trainDataTensor =
      TensorHelpers.fromMat(trainData.filterIx(_ != "label").toMat, device,precision)
    val trainTarget = ATen.squeeze_0(
      TensorHelpers.fromLongMat(
        Mat(trainData.firstCol("label").toVec.map(_.toLong)),
        device
      )
    )
```

## Specify the model

Here we use the predefined modules in `lamp.nn`. 

In particular `lamp.nn.MLP` will create a multilayer
fully connected feed forward neural network. We have to specify the input dimension - in our case 784,
the output dimension - in our case 10, and the sizes of the hidden layers. 
We also have to give it an `aten.TensorOptions` which holds information of what kind of tensor it should allocate for the parameters - what type (float/double/long) and on which device (cpu/cuda).

Then we compose that function with a softmax, and provide a suitable loss function. 

Loss functions in lamp are of the type `(lamp.autograd.Variable, aten.Tensor) => lamp.autograd.Variable`. The first `Variable` argument is the output of the model, the second `Tensor` argument is the target. It returns a new `Variable` which is the loss. An autograd `Variable` holds a tensor and has the ability to compute partial derivatives. The training loop will take the loss and compute the partial derivative of all learnable parameters.

```scala mdoc
import lamp.nn._
val tensorOptions = device.options(SinglePrecision)
val classWeights = ATen.ones(Array(10), tensorOptions)

val model = SupervisedModel(
  sequence(
      MLP(in = 784, out = 10, List(64, 32), tensorOptions, dropout = 0.2),
      Fun(_.logSoftMax(dim = 1))
    ),
  LossFunctions.NLL(10, classWeights)
)
```

## Create mini-batches

The predefined training loop in `lamp.data.IOLoops` needs a stream of batches, where batch holds a subset of the training data.
In particular the trainig loop expects a type of `Unit => lamp.data.BatchStream` 
where the BatchStream represent a stream of batches over the full set of training data. 
This factory function will then be called in each epoch.

Lamp provides a helper which chops up the full batch of data - which we created earlier - 
into appropriately sized mini-batches.
```scala mdoc
import lamp.data._
val makeTrainingBatch = () =>
      BatchStream.minibatchesFromFull(
        minibatchSize = 200,
        dropLast = true,
        features = trainDataTensor,
        target =trainTarget,
        device = device
      )
```

## Create and run the training loop

With this we have everything to assemble the training loop:

```scala mdoc
import cats.effect.IO
val trainedModelIO = IOLoops.epochs(
      model = model,
      optimizerFactory = SGDW
        .factory(
          learningRate = simple(0.0001),
          weightDecay = simple(0.001d)
        ),
      trainBatchesOverEpoch = makeTrainingBatch,
      validationBatchesOverEpoch = None,
      epochs = 1
    )
```

Lamp provides two optimizers: SgdW and AdamW.

The `IOLoop.epochs` method returns an `IO` which will run into the trained model once executed:

```scala mdoc
val trainedModel = trainedModelIO.unsafeRunSync.module
```

The trained model we can use for prediction:
```scala mdoc
val bogusData = ATen.ones(Array(1,784),tensorOptions)
val classProbabilities = trainedModel.forward(const(bogusData)).toMat.map(math.exp)
println(classProbabilities)
```