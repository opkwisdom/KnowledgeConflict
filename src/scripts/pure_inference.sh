#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=$PWD:$PYTHONPATH

# Iterate Map
declare -A DATA_MAP
DATA_MAP["triviaqa"]="validation"
DATA_MAP["truthfulqa"]="validation"
# DATA_MAP["webqa"]="test"

for key in "${!DATA_MAP[@]}"; do
    python3 src/baselines/pure_inference.py \
        --config config/src/baselines/pure_llm.yaml \
        data.name=$key \
        data.data_path="data/$key/retrieved/${DATA_MAP[$key]}.jsonl"
done