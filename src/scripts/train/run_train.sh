#!/bin/bash

export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH=$PWD:$PYTHONPATH

CONFIG_PATH=$1

python3 src/train/run_train.py --config $CONFIG_PATH