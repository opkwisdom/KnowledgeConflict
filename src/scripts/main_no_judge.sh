#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PWD:$PYTHONPATH

CONFIG_PATH=$1

ratios=(0.3)
# lexical_cues=(3 5 7 10)

for ratio in "${ratios[@]}"
do
    python3 src/main_no_judge.py \
        --config $CONFIG_PATH \
        model.prune.ratio=$ratio
done