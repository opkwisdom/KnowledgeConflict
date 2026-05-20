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

ALPHA=1.0
CMODE=split_first
AMODE=full
POOL=max
USE_CAUSAL=False
EPOCHS=2
CKPT_DIR="/workspaces/kvzip_nlplab/checkpoint/stage2_recon_train/multi_LR=0.0001_BS=256_Scratch_Causal=False"

# echo "Sleep for 28 hours to wait reconstruction training..."
# sleep 28h
python3 src/run_gg_clf_train.py \
      --config config/src/train/run_gg_clf_train.yaml \
      data.name=mhqa \
      data.data_path=data/train/mhqa_train_half_force.jsonl \
      train.max_epochs=$EPOCHS \
      train.alpha=$ALPHA \
      caformer.classifier_mode=$CMODE \
      caformer.attention_mode=$AMODE \
      caformer.pooling_strategy=$POOL \
      caformer.use_causal=$USE_CAUSAL \
      caformer.ckpt_dir=$CKPT_DIR

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