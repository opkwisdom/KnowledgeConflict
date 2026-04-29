#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$PWD/src:$PYTHONPATH
export CUDA_LAUNCH_BLOCKING=1
# export TORCH_LOGS="recompiles,graph_breaks"

# Base setting
ALPHAS=(0.0 0.5 1.0)
for ALPHA in "${ALPHAS[@]}"; do
    python3 src/run_gg_train.py \
          --config config/src/train/run_gg_train.yaml \
          data.name=mhqa \
          data.data_path=data/train/mhqa_train_pilot.jsonl \
          train.max_epochs=1 \
          train.alpha=$ALPHA \
          train.gamma=10.0
done
# Convert to oracle mode