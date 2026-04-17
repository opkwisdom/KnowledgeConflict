#!/bin/bash

export CUDA_VISIBLE_DEVICES=3
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$PWD/src:$PYTHONPATH

CONFIG_PATH=$1

python3 src/run_inference.py \
    --config $CONFIG_PATH