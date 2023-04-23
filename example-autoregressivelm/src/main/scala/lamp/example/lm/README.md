# Autoregressive language model

This projects implements a GPT-2 like autoregressive language model.

### Training data 
It trains on any byte level data, thus for text it needs ASCII encoded data (not Unicode).

### How to build
Use the sbt build from the root of the main repository.

### How to train on a single machine (non-distributed)

`example-autoregressivelm/target/universal/stage/bin/example_autoregressivelm --train-file $MYTEXTFILE --valid-file $MYTEXTFILE2 --gpus 0 --checkpoint checkpoint.file`

### How to train on multiple computers (distributed)

E.g. for two computers:
~~~
# root rank, this will drive the training
# checkpointed model files will be on this machine
example-autoregressivelm/target/universal/stage/bin/example_autoregressivelm \
 --train-file $MYTEXTFILE --valid-file $MYTEXTFILE2  --gpus 0 \
 --checkpoint checkpoint \
 --distributed --nranks 2 --root-address ${IP1} --my-address ${IP0} --rank 0

# follower rank
example-autoregressivelm/target/universal/stage/bin/example_autoregressivelm  \
 --train-file $MYTEXTFILE --valid-file $MYTEXTFILE2 --gpus 0 \
 --distributed --nranks 2 --root-address ${IP0}--my-address ${IP1} --rank 1
 ~~~