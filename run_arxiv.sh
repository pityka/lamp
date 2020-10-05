#!/usr/bin/env bash

TOKEN=$(git config --global --get github.token)

id=$(docker --context vm1 build -q .) && \
echo $id && docker --context vm1 run --mount type=volume,source=data1,destination=/data -v /persisted/user/:/persisted \
--env GITHUB_TOKEN=$TOKEN --gpus all $id /bin/bash -c \
"sbt 'example_arxiv/run --gpu --folder ./'"