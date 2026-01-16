#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PWD:$PYTHONPATH

for psg_num in {0..4}
do
    python3 experiments/kv_cache_extract_raw.py \
        --config config/exp/kv_cache_hypothesis_extract_raw.yaml \
        output_dir=experiments/irr_detector_data \
        data.psg_num=$psg_num
done