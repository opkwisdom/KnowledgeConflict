#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$PWD/src:$PYTHONPATH

# CONFIG_PATH=$1

python3 src/run_rc_train.py \
    --config config/src/train/run_rc_train.yaml