#!/usr/bin/env bash

TOKEN=$(git config --global --get github.token)

id=$(docker --context vm1 build -q .) && \
echo $id && docker --context vm1 run -d --mount type=volume,source=data1,destination=/data \
--env GITHUB_TOKEN=$TOKEN --gpus all $id /bin/bash -c \
"sbt 'example_translation/run --gpu --train-data /data/translation/clean.txt --epochs 100 --learning-rate 0.0001 --single --train-batch 160 --validation-batch 32 --checkpoint-save checkpoint'"