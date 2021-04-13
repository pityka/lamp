---
title: 'How to build and train a model'
weight: 2
---

# Overview

Lamp is organized into the following components:

1. A native tensor library storing multidimensional arrays of numbers in off-heap (main or GPU) memory. This is in the `aten` package and links to the native libtorch shared library via JNI. 
2. A more fluent API and lexical scope based memory management in `lamp.STen`. 
3. An algorithm to compute partial derivatives of composite functions. In particular Lamp implements generic reverse mode automatic differentiation. This lives in the package `lamp.autograd`.
5. A set of building blocks to build neural networks in the package `lamp.nn`.
6. Training loop and utilities to work with various kinds of data in `lamp.data`.

# N-dimensional arrays for numeric code

Lamp has native CPU and GPU backed n-dimension arrays with similar operations to numpy or pytorch. 
In fact, `lamp.STen` is a Scala binding to torch's `ATen` tensor class.

```scala mdoc 
import lamp._
// get a scope for zoned memory management
Scope.root{ implicit scope =>

  // a 3D tensor, e.g. a color image
  val img : STen = STen.rand(List(768, 1024, 3))

  // get its shape
  assert(img.shape == List(768, 1024, 3))

  // select a channel
  assert(img.select(dim=2,index=0).shape == List(768, 1024))

  // manipulate with a broadcasting operation
  val img2 = img / 2d

  // take max, returns a tensor with 0 dimensions i.e. a scalar
  assert(img2.max.shape == Nil)

  // get a handle to metadata about data type, data layout and device
  assert(img.options.isCPU)

  val vec = STen.fromVec(org.saddle.Vec(2d,1d,3d))

  // broadcasting matrix multiplication
  val singleChannel = (img matmul vec)
  assert(singleChannel.shape == List(768L,1024L))

  // svd
  val (u,s,v) = singleChannel.svd
  assert(u.shape == List(768,768))
  assert(s.shape == List(768))
  assert(v.shape == List(1024,768))

  val errorOfSVD = (singleChannel - ((u * s) matmul v.t)).norm2(dim=List(0,1), keepDim=false)
  assert(errorOfSVD.toMat.raw(0) < 1E-6)
} 



```

# Building a classifier

We will use tabular data to build a multiclass classifier.

## Get some data

First get some tabular data into the JVM memory:

```scala mdoc:reset
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
      .toOption
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
      .toOption
      .get
```

Next we copy those JVM objects into native tensors. Note that we specify device so it will end up on the desired device.
The feature matrix is copied into a 2D floating point tensor and the target vector is holding the class labels is copied into a 1D integer (long) tensor.
```scala mdoc
    import lamp._
    import lamp.autograd._
    import org.saddle.Mat
    implicit val scope = Scope.free // Use Scope.root, Scope.apply in non-doc code
    val device = CPU
    val precision = SinglePrecision
    val testDataTensor =
      STen.fromMat(testData.filterIx(_ != "label").toMat, device, precision)
    val testTarget = 
      STen.fromLongMat(
        Mat(testData.firstCol("label").toVec.map(_.toLong)),
        device
      ).squeeze
    

    val trainDataTensor =
      STen.fromMat(trainData.filterIx(_ != "label").toMat, device,precision)
    val trainTarget = 
      STen.fromLongMat(
        Mat(trainData.firstCol("label").toVec.map(_.toLong)),
        device
      ).squeeze
    
```

## Specify the model

Here we use the predefined modules in `lamp.nn`. 

In particular `lamp.nn.MLP` will create a multilayer
fully connected feed forward neural network. We have to specify the input dimension - in our case 784,
the output dimension - in our case 10, and the sizes of the hidden layers. 
We also have to give it an `aten.TensorOptions` which holds information of what kind of tensor it should allocate for the parameters - what type (float/double/long) and on which device (cpu/cuda).

Then we compose that function with a softmax, and provide a suitable loss function. 

Loss functions in lamp are of the type `(lamp.autograd.Variable, lamp.STen) => lamp.autograd.Variable`. The first `Variable` argument is the output of the model, the second `STen` argument is the target. It returns a new `Variable` which is the loss. An autograd `Variable` holds a tensor and has the ability to compute partial derivatives. The training loop will take the loss and compute the partial derivative of all learnable parameters.



```scala mdoc
import lamp.nn._
val tensorOptions = device.options(SinglePrecision)
val classWeights = STen.ones(List(10), tensorOptions)
val model = SupervisedModel(
  sequence(
      MLP(in = 784, out = 10, List(64, 32), tensorOptions, dropout = 0.2),
      Fun(implicit scope => _.logSoftMax(dim = 1))
    ),
  LossFunctions.NLL(10, classWeights)
)
```

## Create mini-batches

The predefined training loop in `lamp.data.IOLoops` needs a stream of batches, where batch holds a subset of the training data.
In particular the trainig loop expects a type of `Unit => lamp.data.BatchStream` 
where the BatchStream represent a stream of batches over the full set of training data. 
This factory function will then be called in each epoch.

Lamp provides a helper which chops up the full batch of data into mini-batches.
```scala mdoc
import lamp.data._
val makeTrainingBatch = () =>
      BatchStream.minibatchesFromFull(
        minibatchSize = 200,
        dropLast = true,
        features = trainDataTensor,
        target =trainTarget,
        rng = org.saddle.spire.random.rng.Cmwc5.apply()
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

Lamp provides two optimizers: SgdW and AdamW. They can take a learning rate schedule as well.
For other capabilities of this training loop see the scaladoc of `IOLoops.epochs`.

The `IOLoop.epochs` method returns an `IO` which will run into the trained model once executed:

```scala mdoc
import cats.effect.unsafe.implicits.global
val (epochOfModel, trainedModel, learningCurve) = trainedModelIO.unsafeRunSync()
val module = trainedModel.module
```

The trained model we can use for prediction:
```scala mdoc
val bogusData = STen.ones(List(1,784),tensorOptions)
val classProbabilities = module.forward(const(bogusData)).toMat.map(math.exp)
println(classProbabilities)
```

