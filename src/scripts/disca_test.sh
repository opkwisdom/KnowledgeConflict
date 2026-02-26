#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PWD:$PYTHONPATH

CONFIG_PATH=$1

ratios=(0.3)
python3 src/tests/disca_test.py \
    --config $CONFIG_PATH