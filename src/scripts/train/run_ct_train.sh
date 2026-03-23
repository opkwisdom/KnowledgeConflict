#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$PWD/src:$PYTHONPATH

CONFIG_PATH=$1

python3 src/run_ct_train.py \
    --config $CONFIG_PATH