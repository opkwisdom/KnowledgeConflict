#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=$PWD:$PYTHONPATH

CONFIG_PATH=$1

# Iterate Map
declare -A DATA_MAP
# DATA_MAP["triviaqa"]="validation"
# DATA_MAP["truthfulqa"]="validation"
# DATA_MAP["webqa"]="test"

# for key in "${!DATA_MAP[@]}"; do
#     python3 src/baselines/rag_inference.py \
#         --config $CONFIG_PATH \
#         data.name=$key \
#         data.data_path="data/$key/retrieved/${DATA_MAP[$key]}.jsonl"
# done

python3 src/baselines/rag_inference.py \
    --config $CONFIG_PATH