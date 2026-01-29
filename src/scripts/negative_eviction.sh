#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=$PWD:$PYTHONPATH

CONFIG_PATH=$1

ratios=(0.5 0.9)
prompt_name=sce_modified

for ratio in "${ratios[@]}"
do
    python3 src/negative_eviction.py \
        --config $CONFIG_PATH \
        model.prune.ratio=$ratio \
        judger.prompt_name=$prompt_name
done