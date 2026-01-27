#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# PROJ_ROOT="/workspaces/kvzip_nlplab/KFC-dev"
export PYTHONPATH=$PWD:$PYTHONPATH

CONFIG_PATH=$1

python3 src/train/extract_train_features.py --config $CONFIG_PATH