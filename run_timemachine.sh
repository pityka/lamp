#!/usr/bin/env bash

TOKEN=$(git config --global --get github.token)

id=$(docker --context vm1 build -q .) && echo $id && docker --context vm1 run --mount type=volume,source=data1,destination=/data --env GITHUB_TOKEN=$TOKEN --gpus all $id /bin/bash -c "sbt 'example_timemachine/run --gpu --train-data /data/time_machine/train.txt --test-data /data/time_machine/test.txt --epochs 500 --learning-rate 0.0001 --batch 1024'"