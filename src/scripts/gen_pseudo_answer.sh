#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$PWD/src:$PYTHONPATH

DATASET_NAMES=(2wiki hotpotqa-w)
SPLITS=(train validation test)
for DATASET_NAME in "${DATASET_NAMES[@]}"; do
    for SPLIT in "${SPLITS[@]}"; do
        python src/gen_pseudo_answer.py $DATASET_NAME $SPLIT
    done
done