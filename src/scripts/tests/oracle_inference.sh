#!/bin/bash

export CUDA_VISIBLE_DEVICES=3
export PYTHONPATH=$PWD/src:$PYTHONPATH

python3 src/tests/oracle_inference.py \
    --config config/src/tests/oracle_inference.yaml \
    data.score_mode=topk_10

python3 src/tests/oracle_inference.py \
    --config config/src/tests/oracle_inference.yaml \
    data.score_mode=topk-qa_10