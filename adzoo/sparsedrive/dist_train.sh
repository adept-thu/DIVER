#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-28512}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
TORCH_DISTRIBUTED_DEBUG=DETAIL python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3} --deterministic 
