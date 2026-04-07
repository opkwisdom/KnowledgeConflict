#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$PWD/src:$PYTHONPATH


torchrun --nproc_per_node=3 src/gen_precompute_table.py \
    --config config/src/gen_precompute_table.yaml