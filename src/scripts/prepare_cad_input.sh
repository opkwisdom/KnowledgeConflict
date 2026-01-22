#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PWD:$PYTHONPATH

# Iterate Map
declare -A DATA_MAP
DATA_MAP["triviaqa"]="validation"
DATA_MAP["truthfulqa"]="validation"
DATA_MAP["webqa"]="test"

for key in "${!DATA_MAP[@]}"; do
    python3 src/baselines/context-aware-decoding/prepare_cad_input.py \
        --config config/src/baselines/prepare_cad_input.yaml \
        data.name=$key \
        data.data_path="data/$key/retrieved/${DATA_MAP[$key]}.jsonl" \
        output_dir="src/baselines/context-aware-decoding/eval/${key}_input"
done