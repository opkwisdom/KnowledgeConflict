#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PWD:$PYTHONPATH

python3 src/baselines/context-aware-decoding/prepare_cad_input.py \
    --config config/src/baselines/prepare_cad_input.yaml