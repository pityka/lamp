---
title: 'Training on multiple devices'
weight: 5
---

Lamp can utilize multiple GPUs for training. 
The algorithm lamp uses is a simple version data parallel stochastic gradient descent.
One of the GPUs holds the optimizer and all GPUs hold an identical copy of the model.
All GPUs perform gradient calculations on separate streams of batches of data. 
After each N batches the gradients are transferred and summed up on the GPU which holds the optimizer,
the optimizer updates the model parameters, which in turn are redistributed to the devices.

There is no support for very large models which would need to be striped over multiple devices. 

Lamp supports both distributed and non-distributed multi-GPU training. 
In the distributed setting each GPU may be located in a different computer and/or driven by a different
JVM process, while in the non-distributed multi-GPU setting there is a single JVM which drives and
coordinates the work of all GPUs. 

# Single-process multi-GPU data parallel training

This is the case when all devices are attached to the same computer and can be driven by the same JVM.
In this case the training loop in `lamp.data.IOLoops.epochs` may take an optional argument `dataParallelModels`. `dataParallelModels` receives a list of additional models each allocated onto a different device. 

This method is very simple to use, one only has to allocate the same model onto multiple devices and 
provide those to the training loop.

```scala mdoc:compile-only
  import lamp._ 
  import lamp.nn._ 
  import lamp.data._ 

 Scope.root { implicit scope =>
      // two devices
      val device = CPU
      val device2 = CudaDevice(0)

      val classWeights = STen.ones(List(10), device.options(SinglePrecision))

      // we allocate the exact same model twice, onto each device

      def logisticRegression(inputDim: Int, outDim: Int, tOpt: STenOptions)(implicit scope: Scope) =
        Seq2(
          Linear(inputDim, outDim, tOpt = tOpt),
          Fun(implicit scope => _.logSoftMax(dim = 1))
        )

      val model1 = SupervisedModel(
        logisticRegression(
          784,
          10,
          device.options(SinglePrecision)
        ),
        LossFunctions.NLL(10, classWeights)
      )
      val model2 = SupervisedModel(
        logisticRegression(
          784,
          10,
          device2.options(SinglePrecision)
        ),
        LossFunctions.NLL(10, device2.to(classWeights))
      )


      // We provide both models to the training loop

      IOLoops
        .epochs(
          // provide the training loop the first model as usual in `model`
          // this is where the optimizer will run
          model = model1,
          dataParallelModels = List(model2),
          // rest of the arguments are as in single GPU training
          optimizerFactory = ???,
          trainBatchesOverEpoch = ???,
          validationBatchesOverEpoch = ???,
          epochs = ???,
        )
        
        
        ()
        
 }
```

# Distributed data parallel training

In this settings the GPUs are potentially attached to different computers and the whole training
process is distributed across multiple JVM processes (one process per GPU).

For this use case Lamp has a separate training loop in the `lamp.data.distributed` package. 
The main entry points here are the `lamp.data.distributed.driveDistributedTraining` 
and `lamp.data.distributed.followDistributedTraining` methods.

This training loop in `lamp.data.distributed` uses [NCCL](https://github.com/NVIDIA/nccl) for 
device-device transfers therefor all devices  must be CUDA devices (i.e. no host). 
Due to NCCL the transport optimally uses the available hardware (NVLink, fabric adapters etc).
On the other hand NCCL does not support routing over networks thus each compute node must be on the same
private network (ie. no NAT).

In total the following restrictions apply to the distributed training loop:
- the batch streams of each processes must not contain `EmptyBatch` elements and 
- the batch streams of each processes must contain the exact same number of batches.
- each compute node or process must sit on the same private network
- only CUDA devices
If these are not respected then the distributed process will fail with an exception or worse, go to a dead lock.

## Network transports in distributed training

All tensor data is transfered directly between devices by NCCL.
However NCCL itself does not provide implementations for control messages and initial rendez-vous.
One can use MPI, a raw TCP socket or other means of communication for this purpose. 
Lamp abstracts this functionality away in the `lamp.data.distributed.DistributedCommunication` trait
and provides an implementation using Akka in the `lamp-akka` module.
This message channel for control messages is very low throughput, it broadcasts the NCCL unique id,
and a few messages before and after each epoch.

## Distributed training example

In a distributed setting each process has to set up the model, the batch streams and the 
training loop separately. 
One of these processes is driving the training while the rest follows.

There is a working example in the `example-cifar100-distributed` folder in lamp's source tree.
The part of how to set up the training loop is this:

```scala
        // This is an incomplete example, see the full example in the source tree

        // The following is executed on each processes
         if (config.rank == 0) {
            // if the rank of the process is 0, then it will drive the training loop

            // First get a communication object for the control messages and initial rendez-vous
            val comm = new lamp.distributed.akka.AkkaCommunicationServer(
              actorSystem
            )

            // driveDistributedTraining starts the communication cliques and starts the training loop
            distributed.driveDistributedTraining(
              nranks = config.nranks,
              gpu = config.gpu,
              controlCommunication = comm,
              model = model,
              optimizerFactory = AdamW.factory(
                weightDecay = simple(0.00),
                learningRate = simple(config.learningRate)
              ),
              trainBatches = trainBatches,
              validationBatches = validationBatches,
              maxEpochs = config.epochs
            )

          } else {
            // otherwise, if rank is > 0 then the process on which this is executing
            // will follow the driver, i.e. react to the requests of the driver process

            // to join the communication clique of the driver we have to specify the network 
            // coordinates where to find the driver
            val comm = new lamp.distributed.akka.AkkaCommunicationClient(
              actorSystem,
              config.rootAddress,
              config.rootPort,
              "cifar-0",
              600 seconds
            )

            // `followDistributedTraining` will join the clique and participate in the training
            // process
            distributed.followDistributedTraining(
              rank = config.rank,
              nranks = config.nranks,
              gpu = config.gpu,
              controlCommunication = comm,
              model = model,
              trainBatches = trainBatches,
              validationBatches = validationBatches
            )

          }
```          