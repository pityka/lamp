#!/usr/bin/env bash

rsync -av --exclude-from=rsync.exclude.txt . vm2:~/.
docker --context vm2 run --network host --gpus all -it -v /home/ec2-user/:/build pityka/base-ubuntu-libtorch:torch190-jdk17 /bin/bash -c "cd /build && NCCL_DEBUG=INFO example-cifar100-distributed/target/universal/stage/bin/example_cifar100_distributed --train-data /build/test_data/cifar-100-binary/train.bin --test-data /build/test_data/cifar-100-binary/test.bin --gpu 0 --rank 1 --nranks 2 --root-address ROOT_ADDR --my-address MY_ADDR"