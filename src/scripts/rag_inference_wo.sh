#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
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


# Naive Top-5
# python3 src/baselines/rag_inference.py \
#     --config config/src/baselines/rag.yaml \
#     data.topk_per_query=5


# Rerank Top-5
# python3 src/baselines/rag_inference.py \
#     --config config/src/baselines/rag.yaml \
#     data.topk_per_query=5 \
#     data.do_rerank=True


# Naive Top-3
# python3 src/baselines/rag_inference.py \
#     --config config/src/baselines/rag.yaml \
#     data.topk_per_query=3


# Rerank Top-3
# python3 src/baselines/rag_inference.py \
#     --config config/src/baselines/rag.yaml \
#     data.topk_per_query=3 \
#     data.do_rerank=True


# Rerank Top-10 (total 100)
# sleep 30m
# python3 src/baselines/rag_inference.py \
#     --config config/src/baselines/rag.yaml \
#     data.name=hotpotqa-w \
#     data.data_path=data/hotpotqa-w/retrieved/validation_1000r_top100.jsonl \
#     data.topk_per_query=10 \
#     data.do_rerank=False

python3 src/baselines/rag_inference.py \
    --config config/src/baselines/rag.yaml \
    data.name=hotpotqa-w \
    data.data_path=data/hotpotqa-w/retrieved/validation_p_top100_5000.jsonl \
    data.topk_per_query=10 \
    data.do_rerank=False


# python3 src/baselines/rag_inference.py \
#     --config config/src/baselines/rag.yaml \
#     data.name=hotpotqa-w \
#     data.data_path=data/hotpotqa-w/retrieved/validation_1000r_p_top100.jsonl \
#     data.topk_per_query=10 \
#     data.do_rerank=False