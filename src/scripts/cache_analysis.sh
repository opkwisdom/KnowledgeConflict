#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PWD:$PYTHONPATH

CONFIG_PATH=$1

ratios=(0.3)
for ratio in "${ratios[@]}"
do
    python3 src/cache_analysis.py \
        --config $CONFIG_PATH \
        model.prune.ratio=$ratio
done