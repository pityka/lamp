#!/usr/bin/env bash

TOKEN=$(git config --global --get github.token)

id=$(docker --context vm1 build -q .) && echo $id && docker --context vm1 run -d --mount type=volume,source=data1,destination=/data --env GITHUB_TOKEN=$TOKEN --gpus all $id /bin/bash -c "sbt 'example_gan/run --train-data /data/cifar-100-binary/train.bin --gpu --batch-train 256 --epochs 5000'"