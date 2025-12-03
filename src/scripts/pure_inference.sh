#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=$PWD:$PYTHONPATH

python3 src/baselines/pure_inference.py \
    --config config/src/pure_llm.yaml