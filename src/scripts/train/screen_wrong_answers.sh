#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH=$PWD:$PYTHONPATH
export VLLM_LOGGING_LEVEL=ERROR

python3 src/train/screen_wrong_answers.py