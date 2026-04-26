#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$PWD/src:$PYTHONPATH
export CUDA_LAUNCH_BLOCKING=1
# export TORCH_LOGS="recompiles,graph_breaks"

# Base setting
python3 src/run_gg_train.py \
    --config config/src/train/run_gg_train.yaml

# Convert to oracle mode