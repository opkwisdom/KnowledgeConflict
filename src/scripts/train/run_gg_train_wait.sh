#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$PWD/src:$PYTHONPATH
export CUDA_LAUNCH_BLOCKING=1
# export TORCH_LOGS="recompiles,graph_breaks"

# Base setting
# python3 src/run_gg_train.py \
#       --config config/src/train/run_gg_train.yaml \
#       data.name=mhqa \
#       data.data_path=data/train/mhqa_train_pilot.jsonl \
#       train.max_epochs=1 \
#       train.alpha=1.0


python3 src/run_gg_train.py \
      --config config/src/train/run_gg_train.yaml \
      data.name=mhqa \
      data.data_path=data/train/mhqa_train_pilot.jsonl \
      train.max_epochs=1 \
      train.alpha=0.0 \
      train.gamma=10.0

# echo "Waiting for 1 hour..."
# sleep 1h
# ALPHAS=(0.2 0.5 0.8)
# for ALPHA in "${ALPHAS[@]}"
# do
#   python3 src/run_gg_train.py \
#       --config config/src/train/run_gg_train.yaml \
#       data.name=mhqa \
#       data.data_path=data/train/mhqa_train_pilot.jsonl \
#       train.max_epochs=1 \
#       train.alpha=$ALPHA
# done


# Convert to oracle mode