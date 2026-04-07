#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$PWD/src:$PYTHONPATH

echo "Sleep for 7 hours to wait for the training to finish 1 epoch..."
sleep 7h

echo "Kill the training process and start generating precompute table..."
pkill -f "python3 src/run_gg_train.py"
sleep 3m

bash src/scripts/gen_precompute_table.sh