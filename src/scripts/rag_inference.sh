#!/bin/bash

export CUDA_VISIBLE_DEVICES=2
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$PWD:$PYTHONPATH

# Iterate Map
# declare -A DATA_MAP
# DATA_MAP["triviaqa"]="validation"
# DATA_MAP["truthfulqa"]="validation"
# DATA_MAP["webqa"]="test"

# for key in "${!DATA_MAP[@]}"; do
#     python3 src/baselines/rag_inference.py \
#         --config config/src/baselines/rag.yaml \
#         data.name=$key \
#         data.data_path="data/$key/retrieved/${DATA_MAP[$key]}.jsonl"
# done


### RAG config paths
CONFIG_WO_PATH=config/src/baselines/rag_wo.yaml
CONFIG_SMALL_PATH=config/src/baselines/rag_small.yaml
CONFIG_LARGE_PATH=config/src/baselines/rag_large.yaml


python3 src/baselines/rag_inference.py \
    --config config/src/baselines/rag.yaml \
    data.name=hotpotqa-w \
    data.data_path=data/hotpotqa-w/retrieved/validation_p_top100_5000.jsonl \
    data.topk_per_query=10