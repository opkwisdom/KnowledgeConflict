#!/bin/bash

export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH=$PWD:$PYTHONPATH

CONFIG_PATH=$1

ratios=(0.3)
# lexical_cues=(3 5 7 10)
thres_values=(0.9 0.91 0.92 0.93 0.94 0.96 0.97 0.98 0.99)

for ratio in "${ratios[@]}"
do
    for thres in "${thres_values[@]}"
    do
        python3 src/main_no_judge.py \
            --config $CONFIG_PATH \
            model.prune.ratio=$ratio \
            uncertainty_estimator.thres=$thres
    done
done