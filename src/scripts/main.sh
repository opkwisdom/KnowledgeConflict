#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=$PWD:$PYTHONPATH

ratios=(0.1 0.3 0.5 0.7 0.9)
for ratio in "${ratios[@]}"
do
    python3 src/main.py \
        --config config/src/main.yaml \
        model.prune.ratio=$ratio
done