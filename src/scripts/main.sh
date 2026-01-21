#!/bin/bash

export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH=$PWD:$PYTHONPATH

CONFIG_PATH=$1

ratios=(0.3)
lexical_cues=(3 5 7 10 20 30)
for ratio in "${ratios[@]}"
do
    for cues in "${lexical_cues[@]}"
    do
        echo "Running experiment with Ratio: $ratio, Lexical Cues: $cues"

        python3 src/main.py \
            --config $CONFIG_PATH \
            model.prune.ratio=$ratio \
            model.lexical_cue.n_tokens=$cues
    done
done