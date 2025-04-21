#!/usr/bin/env bash

vm=$1
arg=$2
id=$(docker --context $vm build -q .) && echo $id && docker --context $vm run  --gpus all $id /bin/bash -c "sbt/bin/sbt '$arg'"