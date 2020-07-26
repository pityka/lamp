#!/usr/bin/env bash

TOKEN=$(git config --global --get github.token)

id=$(docker --context vm1 build -q .) && echo $id && docker --context vm1 run -d --env GITHUB_TOKEN=$TOKEN --gpus all $id /bin/bash -c "sbt 'testOnly *HousePricesSuite'"