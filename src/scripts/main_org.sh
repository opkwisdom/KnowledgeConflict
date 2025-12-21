#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=$PWD:$PYTHONPATH

CONFIG_PATH=$1

ratios=(0.3)
for ratio in "${ratios[@]}"
do
    python3 src/main_org.py \
        --config $CONFIG_PATH \
        model.prune.ratio=$ratio \
        judger.prompt_name=base
done