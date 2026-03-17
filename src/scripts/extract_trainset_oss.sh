#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH=$PWD:$PYTHONPATH

CONFIG_PATH=$1

python3 src/extract_trainset_oss.py \
    --config $CONFIG_PATH