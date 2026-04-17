#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$PWD/src:$PYTHONPATH

CONFIG_PATH=$1

### Hyperparameter search space ###

# Debugging
# python3 src/run_train.py \
#     --config $CONFIG_PATH

# Exp 1: Do not use freeze_pretrained
# echo "Starting Exp 1: freeze_pretrained=False at $(date)"
# python3 src/run_train.py \
#     --config $CONFIG_PATH \
#     train.freeze_pretrained=False
# echo "Finished Exp 1: freeze_pretrained=False at $(date)"

# Exp 2: Use freeze_pretrained
echo "Starting Exp 2: freeze_pretrained=True at $(date)"
python3 src/run_train.py \
    --config $CONFIG_PATH \
    train.freeze_pretrained=True
echo "Finished Exp 2: freeze_pretrained=True at $(date)"