#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$PWD/src:$PYTHONPATH

CONFIG_PATH=$1

# Base setting
python3 src/run_gg_train.py \
    --config config/src/train/run_gg_train.yaml

# Convert to oracle mode