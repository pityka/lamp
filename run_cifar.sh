#!/usr/bin/env bash

TOKEN=$(git config --global --get github.token)

id=$(docker --context vm1 build -q .) && echo $id && docker --context vm1 run --mount type=volume,source=data1,destination=/data --env GITHUB_TOKEN=$TOKEN --gpus all $id /bin/bash -c "sbt 'example_cifar100/run --train-data /data/cifar-100-binary/train.bin --test-data /data/cifar-100-binary/test.bin --label-data /data/cifar-100-binary/fine_label_names.txt --gpu'"