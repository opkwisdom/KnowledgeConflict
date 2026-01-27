#!/bin/bash

export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH=$PWD:$PWD/KVzip:$PYTHONPATH

CONFIG_PATH=$1

# ratios=(0.3)
# lexical_cues=(3 5 7 10)
declare -A datasets
datasets["nq"]="data/nq/parametric_relevance_tagged/validation.json"
# datasets["triviaqa"]="data/triviaqa/retrieved/validation.jsonl"
# datasets["truthfulqa"]="data/truthfulqa/retrieved/validation.jsonl"

# top_ks=(1 2 3 5 7 10 20 30 50 100)
top_ks=(2)

for name in "${!datasets[@]}"
do
    path=${datasets[$name]}
    
    echo "========================================================"
    echo "Running experiment for dataset: $name"
    echo "Data path: $path"
    echo "========================================================"

    for top_k in "${top_ks[@]}"
    do
        python3 src/boosted_prob_toy.py \
            --config $CONFIG_PATH \
            data.name=$name \
            data.data_path=$path \
            uncertainty_estimator.top_k=$top_k
    done
done