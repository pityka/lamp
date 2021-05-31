#!/usr/bin/env bash

TOKEN=$(git config --global --get github.token)



sbt example_cifar100/stage && cp Dockerfile example-cifar100/target/universal/stage/ && cd example-cifar100/target/universal/stage/ && id=$(docker --context vm1 build -q .) && echo $id && docker --context vm1 run --ulimit memlock=81920000000:81920000000 --mount type=volume,source=data1,destination=/data --env GITHUB_TOKEN=$TOKEN --gpus all $id /bin/bash -c "ulimit -l ; nsys profile --delay=30 --duration=10 --kill=none bin/example_cifar100 --train-data /data/cifar-100-binary/train.bin --test-data /data/cifar-100-binary/test.bin --label-data /data/cifar-100-binary/fine_label_names.txt --gpus 0,1,2,3 --learning-rate 0.001 --network resnet --checkpoint-save checkpoint.cifar --batch-train 2048 --batch-test 256 --single --pinned --epochs 1000 --dropout 0.5 "