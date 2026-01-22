#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PWD:$PYTHONPATH

CONFIG_PATH=$1

ratios=(0.3)
prompt_name=sce_modified

for ratio in "${ratios[@]}"
do
    python3 src/main.py \
        --config $CONFIG_PATH \
        model.prune.ratio=$ratio \
        judger.prompt_name=$prompt_name
done