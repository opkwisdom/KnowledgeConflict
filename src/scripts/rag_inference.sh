#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=$PWD:$PYTHONPATH

python3 src/baselines/rag_inference.py \
    --config config/src/baselines/rag.yaml