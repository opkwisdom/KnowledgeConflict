#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PWD:$PYTHONPATH

python3 src/baselines/pure_inference.py \
    --config config/src/baselines/pure_llm.yaml